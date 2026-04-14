# -*- coding: utf-8 -*-
"""
PNMS 主系统：编排查询处理流程（文档 §10.1）、冷启动（§12）、各组件调用。
与 LLM 解耦：通过回调或接口注入“推理器”，本模块只负责记忆检索、上下文构建与记忆更新。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import PNMSConfig
from .state import PersonalNeuralState
from .memory import MemoryStore
from .graph import MemoryGraph
from .concept import ConceptModuleManager
from .context import ContextBuilder
from .encoder import SimpleQueryEncoder, SentenceEncoder
from .slot import MemorySlot
from .exceptions import PersistenceError, LLMError, ConfigError, ErrorCodes
from .versioning import (
    MEMORY_FORMAT_VERSION,
    is_backward_compatible_checkpoint,
    parse_semver,
    peek_checkpoint_versions,
)
from .update import (
    compute_interaction_embedding,
    try_form_concept_modules_from_store,
    update_state,
    update_memory,
)


logger = logging.getLogger("pnms.system")

# 类型：接收 (query, context_str) 返回 response 文本
LLMReasoner = Callable[[str, str], str]


class PNMS:
    """
    个人神经记忆系统主类（文档 §4、§10）。
    流程：Query -> 编码 -> 记忆检索（含冷启动判断）-> 图扩展 -> 概念模块（可选）-> 上下文构建 -> LLM -> 记忆更新。
    """

    def __init__(
        self,
        config: Optional[PNMSConfig] = None,
        user_id: str = "default",
        encoder: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or PNMSConfig()
        # 启动时做一次配置校验，尽早暴露错误
        try:
            self.config.validate()
        except AttributeError:
            # 兼容旧版本未提供 validate 的场景
            pass
        self.user_id = user_id
        self.device = device or torch.device("cpu")
        # 编码器：优先使用语义编码（sentence-transformers），不可用时回退到字符级简单编码
        if encoder is not None:
            self.encoder = encoder
        else:
            try:
                self.encoder = SentenceEncoder(device=self.device)
            except ImportError:
                self.encoder = SimpleQueryEncoder(
                    self.config.embed_dim,
                    vocab_size=10000,
                ).to(self.device)
        if hasattr(self.encoder, "embed_dim"):
            self.config.embed_dim = self.encoder.embed_dim
        # 个人神经状态 S_t
        self.state = PersonalNeuralState(
            self.config.embed_dim,
            alpha=self.config.state_alpha,
            device=self.device,
        )
        # 记忆图 G
        self.graph = MemoryGraph(
            cooccur_delta=self.config.graph_cooccur_delta,
            edge_decay=self.config.graph_edge_decay,
        )
        # 记忆存储 M（含槽与检索）
        self.store = MemoryStore(self.config, self.user_id, graph=self.graph)
        # 概念模块管理器 C（可选）；若配置了 concept_checkpoint_dir 则尝试加载已保存模块
        self.concept_manager = ConceptModuleManager(
            self.config.embed_dim,
            concept_dim=self.config.concept_dim,
            top_m=self.config.concept_top_m,
        )
        if getattr(self.config, "concept_checkpoint_dir", None):
            ckpt = Path(self.config.concept_checkpoint_dir)
            if ckpt.is_dir():
                if (ckpt / "meta.json").exists():
                    try:
                        self.concept_manager.load(ckpt, device=self.device)
                    except (ValueError, FileNotFoundError):
                        pass
                for graph_path in (ckpt / "graph.db", ckpt / "graph.json"):
                    if graph_path.exists():
                        try:
                            self.graph.load(graph_path)
                            break
                        except Exception:
                            pass
        # 上下文构建器（§10.3 L_max token 预算）
        self.context_builder = ContextBuilder(
            max_context_slots=self.config.max_context_slots,
            max_context_tokens=self.config.max_context_tokens_estimate,
        )
        # §11.4 每 N 轮触发衰减；§9.2 概念形成间隔
        self._round_counter = 0
        if getattr(self.config, "concept_checkpoint_dir", None):
            self._try_load_memory_snapshot(Path(self.config.concept_checkpoint_dir))
        logger.info(
            "PNMS initialized for user=%s embed_dim=%d concept_enabled=%s graph_enabled=%s slots=%d",
            self.user_id,
            self.config.embed_dim,
            self.config.concept_enabled,
            self.config.graph_enabled,
            self.store.num_slots,
        )

    def encode(self, text: str) -> torch.Tensor:
        """
        将文本编码为向量 q。若编码器支持 encode_text 则走语义编码，否则走字符级编码。
        """
        if hasattr(self.encoder, "encode_text"):
            with torch.no_grad():
                q = self.encoder.encode_text(text)
            return q.to(self.device)
        ids = [min(ord(c) % 10000, 9999) for c in text[:256]]
        if not ids:
            ids = [0]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            q = self.encoder(x)
        return q.squeeze(0)

    def handle_query(
        self,
        query: str,
        llm: LLMReasoner,
        response: Optional[str] = None,
        content_to_remember: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        处理单轮查询（文档 §10.1 主流程）。

        Args:
            query: 用户输入文本。
            llm: 回调 (query, context_str) -> response 文本。
            response: 若已由外部生成 response，可传入以省去调用 llm，仅做记忆更新。
            content_to_remember: 本轮要写入记忆的内容（若为 None 可由 response 摘要代替或留空）。
            system_prompt: 可选系统提示。

        Returns:
            回复文本。
        """
        # 1. 编码
        q = self.encode(query)
        # 冷启动阶段决定是否检索、是否图扩展、是否概念模块（文档 §12.2）
        phase = self.config.in_cold_start_phase(self.store.num_slots)

        # 2. 记忆检索：phase=="pure_llm" 时不查槽，仅积累 S 与写槽
        slots: List[MemorySlot] = []
        if phase == "pure_llm":
            # 不检索，只更新状态与写槽（若有）
            pass
        else:
            pairs = self.store.retrieve(
                q,
                top_k=self.config.retrieval_top_k,
                state_vector=self.state.state,
            )
            slots = [s for s, _ in pairs]
            if self.config.min_slots_to_use_memory and len(slots) < self.config.min_slots_to_use_memory:
                slots = []

            # 3. 图扩展（仅 slots_and_graph 且启用图时）
            if (
                phase == "slots_and_graph"
                and self.config.graph_enabled
                and slots
            ):
                slot_ids = [s.slot_id for s in slots]
                expanded_ids = self.graph.expand_slots(
                    slot_ids,
                    self.store.slot_by_id,
                    self.config.graph_max_neighbors_per_slot,
                    self.config.graph_max_expanded_total,
                )
                slots = self.store.get_slots_by_ids(expanded_ids)

            # 4. 概念模块 Augment（全量阶段且启用概念时）
            if (
                phase == "slots_and_graph"
                and self.config.concept_enabled
                and self.concept_manager._modules
                and slots
            ):
                mods = self.concept_manager.retrieve_modules(q, self.config.concept_top_m)
                if mods:
                    mod_ids = [m[0] for m in mods]
                    slots = self.concept_manager.augment_slots(
                        slots, q, self.state.state, mod_ids
                    )

        # 5. 上下文构建（§12.3 槽不足时可选 RAG/知识库补足）
        state_summary = None
        context_supplement = None
        if (len(slots) < self.config.min_slots_to_use_memory or not slots) and self.config.context_supplement:
            context_supplement = self.config.context_supplement(
                query, len(slots), self.config.max_context_slots
            )
        context_str = self.context_builder.build(
            slots,
            state_summary=state_summary,
            system_prompt=system_prompt,
            context_supplement=context_supplement,
        )

        # 6. LLM 推理（若未提供 response 则调用 llm）
        if response is None:
            start = time.monotonic()
            try:
                response = llm(query, context_str)
                elapsed = time.monotonic() - start
                logger.info(
                    "LLM call succeeded for user=%s (len(query)=%d, context_chars=%d, elapsed=%.3fs)",
                    self.user_id,
                    len(query),
                    len(context_str),
                    elapsed,
                )
            except Exception as e:
                elapsed = time.monotonic() - start
                logger.exception(
                    "LLM call failed for user=%s (len(query)=%d, context_chars=%d, elapsed=%.3fs)",
                    self.user_id,
                    len(query),
                    len(context_str),
                    elapsed,
                )
                raise LLMError("LLM call failed", cause=e)

        # 7. 记忆更新
        response_embed = self.encode(response) if response else None
        E_t = compute_interaction_embedding(q, response_embed, agg="mean")
        update_state(self.state, E_t)
        to_write = content_to_remember or (response[:500] if response else None)
        activated_ids = [s.slot_id for s in slots]
        update_memory(
            self.config,
            self.store,
            self.graph if phase != "pure_llm" else None,
            q,
            response_embed,
            to_write,
            activated_ids,
        )
        self._round_counter += 1
        # §11.4 每 N 轮触发槽权重衰减与图边衰减
        if self.config.decay_every_n_rounds and self._round_counter % self.config.decay_every_n_rounds == 0:
            self.store.apply_decay()
            self.graph.decay_edges()
        # §9.2 周期性尝试概念形成（满足 N_min、slots_and_graph 且尚无模块时）
        if (
            self.config.concept_formation_interval_rounds
            and self._round_counter % self.config.concept_formation_interval_rounds == 0
        ):
            formed = try_form_concept_modules_from_store(
                self.config,
                self.store,
                self.concept_manager,
                self.state,
                self.device,
            )
            if formed > 0 and getattr(self.config, "concept_checkpoint_dir", None):
                self.save_concept_modules()

        return response

    def _save_memory_snapshot(self, root: Path) -> None:
        """记忆槽与个人状态 S_t、轮次计数；与概念 checkpoint 同目录。"""
        root.mkdir(parents=True, exist_ok=True)
        slots_path = root / "memory_slots.json"
        payload = {
            "version": 1,
            "user_id": self.user_id,
            "slots": self.store.export_slots_json(),
        }
        slots_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        session_path = root / "memory_session.pt"
        torch.save(
            {
                "version": 1,
                "state": self.state.state.detach().cpu(),
                "state_initialized": self.state.is_initialized,
                "round_counter": self._round_counter,
            },
            session_path,
        )
        logger.debug(
            "saved memory snapshot: %d slots, round=%d into %s",
            self.store.num_slots,
            self._round_counter,
            root,
        )

    def _try_load_memory_snapshot(self, root: Path) -> None:
        """从 ``memory_slots.json`` + ``memory_session.pt`` 恢复（若无文件则跳过）。"""
        slots_path = root / "memory_slots.json"
        if not slots_path.is_file():
            return
        try:
            raw = json.loads(slots_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("failed to read memory_slots.json from %s", root)
            return
        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict) and isinstance(raw.get("slots"), list):
            rows = raw["slots"]
        else:
            logger.warning("memory_slots.json format not recognized under %s", root)
            return
        try:
            self.store.clear_and_load_from_json(rows, device=self.device)
        except Exception:
            logger.exception("failed to apply memory_slots.json for user=%s", self.user_id)
            return
        sess = root / "memory_session.pt"
        if sess.is_file():
            try:
                try:
                    data = torch.load(sess, map_location=self.device, weights_only=True)
                except TypeError:
                    data = torch.load(sess, map_location=self.device)
            except Exception:
                logger.exception("failed to load memory_session.pt for user=%s", self.user_id)
                return
            if isinstance(data, dict) and "state" in data:
                st = data["state"]
                if isinstance(st, torch.Tensor):
                    try:
                        self.state.load_from_tensor(
                            st,
                            initialized=bool(data.get("state_initialized", True)),
                        )
                    except ValueError as e:
                        logger.warning("skip neural state load: %s", e)
                if "round_counter" in data:
                    self._round_counter = int(data["round_counter"])
        logger.info(
            "loaded memory snapshot for user=%s: slots=%d round=%d",
            self.user_id,
            self.store.num_slots,
            self._round_counter,
        )

    def save_concept_modules(self, path: Optional[str] = None) -> None:
        """
        将当前已训练的概念神经网络与记忆图边表保存到目录（§13.3、§8）。
        若未传 path，则使用 config.concept_checkpoint_dir；未设置则不做任何操作。
        写入：概念 meta.json + 各 module_id.pt；图链接 graph.db（SQLite）；
        以及记忆槽 ``memory_slots.json``、会话状态 ``memory_session.pt``。
        """
        dir_path = path or getattr(self.config, "concept_checkpoint_dir", None)
        if not dir_path:
            return
        root = Path(dir_path)
        try:
            self.concept_manager.save(root)
            self.graph.save(root / "graph.db")
            self._save_memory_snapshot(root)
            logger.info(
                "saved concepts, graph and memory snapshot for user=%s into %s",
                self.user_id,
                root,
            )
        except PersistenceError:
            # 透传给调用方，同时保留日志
            logger.exception("failed to save concepts/graph for user=%s", self.user_id)
            raise

    def load_concept_modules(
        self,
        path: Optional[str] = None,
        expected_memory_format_version: str = MEMORY_FORMAT_VERSION,
    ) -> None:
        """
        从目录加载已保存的概念模块与图边表，覆盖当前 concept_manager 与 graph。
        若未传 path，则使用 config.concept_checkpoint_dir。
        加载前会先做记忆文件格式版本校验：只允许向下兼容（checkpoint_version <= expected_memory_format_version）。
        """
        dir_path = path or getattr(self.config, "concept_checkpoint_dir", None)
        if not dir_path:
            return
        root = Path(dir_path)
        try:
            parse_semver(expected_memory_format_version)
        except ValueError as e:
            raise ConfigError(
                "expected_memory_format_version must be semver like major.minor.patch",
                cause=e,
            )

        vinfo = peek_checkpoint_versions(root)
        meta_v = vinfo.get("memory_format_in_meta")
        graph_v = vinfo.get("memory_format_in_graph")
        if meta_v and graph_v and meta_v != graph_v:
            raise PersistenceError(
                f"checkpoint format version mismatch: meta.json={meta_v}, graph.db={graph_v}"
            )
        checkpoint_v = meta_v or graph_v
        if checkpoint_v:
            try:
                if not is_backward_compatible_checkpoint(checkpoint_v, expected_memory_format_version):
                    raise PersistenceError(
                        "checkpoint memory format is newer than supported: "
                        f"checkpoint={checkpoint_v}, expected={expected_memory_format_version}, "
                        "PNMS only guarantees backward compatibility (not forward compatibility)"
                    )
            except ValueError as e:
                raise PersistenceError(
                    f"invalid checkpoint memory format version: {checkpoint_v}",
                    cause=e,
                )
        try:
            self.concept_manager.load(root, device=self.device)
            graph_path = root / "graph.db"
            if graph_path.exists():
                try:
                    self.graph.load(graph_path)
                except PersistenceError:
                    logger.exception(
                        "failed to load graph for user=%s from %s", self.user_id, graph_path
                    )
                    raise
            self._try_load_memory_snapshot(root)
            logger.info(
                "loaded concepts, graph and memory snapshot for user=%s from %s",
                self.user_id,
                root,
            )
        except Exception:
            # 不让初始化直接崩溃，由调用方按需重试或忽略；日志中保留堆栈
            logger.exception(
                "error while loading concepts/graph for user=%s from %s",
                self.user_id,
                root,
            )

    def get_context_for_query(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        use_concept: Optional[bool] = None,
    ) -> Tuple[str, int]:
        """
        只读路径：对 query 做编码、检索、图扩展、可选概念 Augment、上下文构建，返回将发送给 LLM 的提示词。
        不更新状态、不写槽，用于对比「概念启用前/后」的提示差异与是否压缩。
        use_concept: None=按当前 config 与 _modules；True=强制走概念 Augment；False=强制不走概念。
        """
        q = self.encode(query)
        phase = self.config.in_cold_start_phase(self.store.num_slots)
        slots: List[MemorySlot] = []
        if phase != "pure_llm":
            pairs = self.store.retrieve(
                q,
                top_k=self.config.retrieval_top_k,
                state_vector=self.state.state,
            )
            slots = [s for s, _ in pairs]
            if self.config.min_slots_to_use_memory and len(slots) < self.config.min_slots_to_use_memory:
                slots = []
            if (
                phase == "slots_and_graph"
                and self.config.graph_enabled
                and slots
            ):
                slot_ids = [s.slot_id for s in slots]
                expanded_ids = self.graph.expand_slots(
                    slot_ids,
                    self.store.slot_by_id,
                    self.config.graph_max_neighbors_per_slot,
                    self.config.graph_max_expanded_total,
                )
                slots = self.store.get_slots_by_ids(expanded_ids)
            do_concept = use_concept if use_concept is not None else (
                self.config.concept_enabled and bool(self.concept_manager._modules)
            )
            if (
                phase == "slots_and_graph"
                and do_concept
                and self.concept_manager._modules
                and slots
            ):
                mods = self.concept_manager.retrieve_modules(q, self.config.concept_top_m)
                if mods:
                    mod_ids = [m[0] for m in mods]
                    slots = self.concept_manager.augment_slots(
                        slots, q, self.state.state, mod_ids
                    )
        state_summary = None
        context_supplement = None
        if (len(slots) < self.config.min_slots_to_use_memory or not slots) and self.config.context_supplement:
            context_supplement = self.config.context_supplement(
                query, len(slots), self.config.max_context_slots
            )
        context_str = self.context_builder.build(
            slots,
            state_summary=state_summary,
            system_prompt=system_prompt,
            context_supplement=context_supplement,
        )
        return context_str, len(slots)

    def merge_memories(
        self,
        source_checkpoint_dir: str,
        source_memory_format_version: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        将「外部 checkpoint」中的记忆槽并入当前运行中的引擎（简化实现）。

        当前策略是“粗暴可用版”：仅复制源 ``memory_slots.json`` 的槽到当前存储，
        每条槽通过 ``MemoryStore.write`` 写入（会按现有阈值走新增/更新/merge）。
        不合并 ``graph.db``、概念模块、``memory_session.pt``。

        版本门禁：
        - 合并源版本必须 <= 当前运行版本 ``MEMORY_FORMAT_VERSION``（仅向下兼容，不向上兼容）。
        - ``source_memory_format_version`` 为空时，自动从 ``source_checkpoint_dir`` 中探测。

        Returns:
            合并结果统计，便于调用方拿到明确结果：
            ``{"source_checkpoint_dir", "source_memory_format_version", "merged_slots", "before_slots", "after_slots"}``
        """
        root = Path(source_checkpoint_dir)
        if not root.exists():
            raise PersistenceError(
                f"source checkpoint dir not found: {root}",
                code=ErrorCodes.MERGE_CHECKPOINT_NOT_FOUND,
            )
        if source_memory_format_version is not None:
            try:
                parse_semver(source_memory_format_version)
            except ValueError as e:
                raise ConfigError(
                    "source_memory_format_version must be semver like major.minor.patch",
                    cause=e,
                    code=ErrorCodes.MERGE_INVALID_ARGUMENT,
                )

        detected = peek_checkpoint_versions(root)
        detected_meta = detected.get("memory_format_in_meta")
        detected_graph = detected.get("memory_format_in_graph")
        if detected_meta and detected_graph and detected_meta != detected_graph:
            raise PersistenceError(
                f"source checkpoint format mismatch: meta.json={detected_meta}, graph.db={detected_graph}",
                code=ErrorCodes.MERGE_INVALID_ARGUMENT,
            )
        source_v = source_memory_format_version or detected_meta or detected_graph
        if not source_v:
            raise PersistenceError(
                "source checkpoint memory format version is missing; "
                "please pass source_memory_format_version explicitly",
                code=ErrorCodes.MERGE_INVALID_ARGUMENT,
            )
        try:
            if not is_backward_compatible_checkpoint(source_v, MEMORY_FORMAT_VERSION):
                raise PersistenceError(
                    "source checkpoint memory format is newer than current runtime: "
                    f"source={source_v}, current={MEMORY_FORMAT_VERSION}",
                    code=ErrorCodes.MERGE_VERSION_INCOMPATIBLE,
                )
        except ValueError as e:
            raise PersistenceError(
                f"invalid source checkpoint memory format version: {source_v}",
                cause=e,
                code=ErrorCodes.MERGE_INVALID_ARGUMENT,
            )

        slots_path = root / "memory_slots.json"
        if not slots_path.is_file():
            before_slots = self.store.num_slots
            return {
                "source_checkpoint_dir": str(root),
                "source_memory_format_version": source_v,
                "merged_slots": 0,
                "before_slots": before_slots,
                "after_slots": before_slots,
            }
        try:
            raw = json.loads(slots_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise PersistenceError(
                f"failed to read source memory_slots.json: {slots_path}",
                cause=e,
                code=ErrorCodes.MERGE_INVALID_ARGUMENT,
            )
        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict) and isinstance(raw.get("slots"), list):
            rows = raw["slots"]
        else:
            raise PersistenceError(
                "source memory_slots.json format not recognized",
                code=ErrorCodes.MERGE_INVALID_ARGUMENT,
            )
        before_slots = self.store.num_slots
        merged_slots = 0
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise PersistenceError(
                    f"source memory_slots.json: slot at index {i} is not an object",
                    code=ErrorCodes.MERGE_INVALID_ARGUMENT,
                )
            try:
                slot = MemorySlot.from_dict(
                    row,
                    embed_dim=self.config.embed_dim,
                    device=self.device,
                    user_id=self.user_id,
                )
                self.store.write(
                    slot.key,
                    slot.content,
                    weight=slot.weight,
                    source=slot.source,
                )
                merged_slots += 1
            except Exception as e:
                raise PersistenceError(
                    f"failed to merge slot at index {i}: {e}",
                    cause=e,
                    code=ErrorCodes.MERGE_INVALID_ARGUMENT,
                ) from e
        after_slots = self.store.num_slots
        logger.info(
            "merge_memories completed for user=%s: merged=%d slots(before=%d, after=%d)",
            self.user_id,
            merged_slots,
            before_slots,
            after_slots,
        )
        return {
            "source_checkpoint_dir": str(root),
            "source_memory_format_version": source_v,
            "merged_slots": merged_slots,
            "before_slots": before_slots,
            "after_slots": after_slots,
        }

    def get_state_summary(self) -> dict:
        """返回当前系统状态摘要，便于实验观测（文档 §5.1 X_t）。"""
        return {
            "user_id": self.user_id,
            "num_slots": self.store.num_slots,
            "phase": self.config.in_cold_start_phase(self.store.num_slots),
            "state_norm": float(self.state.state.norm().item()),
        }
