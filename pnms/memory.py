# -*- coding: utf-8 -*-
"""
记忆存储与检索（文档 §7、§10.2）。
维护用户级记忆槽列表，支持按 query 嵌入的 top-k 检索、写入、淘汰与权重更新。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import PNMSConfig
from .slot import MemorySlot, SlotSource
from .graph import MemoryGraph

logger = logging.getLogger("pnms.memory")


def cosine_similarity(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """sim(q, k_i) = q·k_i / (||q|| ||k_i||)（文档 §7.2）。"""
    q = q.flatten().float()
    k = k.flatten().float()
    nq = q.norm().clamp(min=1e-8)
    nk = k.norm().clamp(min=1e-8)
    return (q @ k) / (nq * nk)


class MemoryStore:
    """
    记忆槽存储 + 检索（文档 §7、§13.1）。
    当前实现：暴力 top-k 检索（O(S*D)）；大规模时可替换为 FAISS/HNSW（§13.4）。
    """

    def __init__(self, config: PNMSConfig, user_id: str, graph: Optional[MemoryGraph] = None):
        self.config = config
        self.user_id = user_id
        self.graph = graph
        self._slots: List[MemorySlot] = []
        self._slot_by_id: Dict[str, MemorySlot] = {}

    def _normalize_key(self, key: torch.Tensor) -> torch.Tensor:
        k = key.detach().float().flatten()
        if k.shape[0] != self.config.embed_dim:
            raise ValueError(f"key dim {k.shape[0]} != embed_dim {self.config.embed_dim}")
        norm = k.norm().clamp(min=1e-8)
        return (k / norm)

    def retrieve(
        self,
        q: torch.Tensor,
        top_k: Optional[int] = None,
        state_vector: Optional[torch.Tensor] = None,
        use_weight: bool = True,
    ) -> List[Tuple[MemorySlot, float]]:
        """
        按 query 嵌入 q 检索 top-k 槽（文档 §7.2）。
        相似度 sim(q, k_i)；可选乘以 weight 重排；可选与 state_vector 结合（§6.4）。
        返回 [(slot, score), ...] 按 score 降序。
        """
        if not self._slots:
            return []
        k = top_k or self.config.retrieval_top_k
        q = self._normalize_key(q)
        if state_vector is not None:
            # 简单融合：q' = q + 0.2 * S_u 再归一化（可调）
            q = self._normalize_key(q + 0.2 * state_vector.flatten().to(q.device))

        scores: List[Tuple[int, float]] = []
        for i, slot in enumerate(self._slots):
            sim = cosine_similarity(q, slot.key.to(q.device)).item()
            if use_weight:
                score = sim * slot.weight
            else:
                score = sim
            scores.append((i, score))
        scores.sort(key=lambda x: -x[1])
        out: List[Tuple[MemorySlot, float]] = []
        for idx, sc in scores[:k]:
            slot = self._slots[idx]
            slot.bump_access(
                weight_bump=self.config.slot_weight_bump_on_access,
                weight_gamma=self.config.slot_access_weight_gamma,
            )
            out.append((slot, sc))
        return out

    def retrieve_slot_ids_and_scores(
        self,
        q: torch.Tensor,
        top_k: Optional[int] = None,
        state_vector: Optional[torch.Tensor] = None,
    ) -> List[Tuple[str, float]]:
        """仅返回 (slot_id, score)，便于图扩展时用 ID 查槽。"""
        pairs = self.retrieve(q, top_k=top_k, state_vector=state_vector)
        return [(s.slot_id, sc) for s, sc in pairs]

    def get_slots_by_ids(self, slot_ids: List[str]) -> List[MemorySlot]:
        """按 ID 列表返回槽（图扩展后去重排序用）。"""
        out = []
        for sid in slot_ids:
            if sid in self._slot_by_id:
                out.append(self._slot_by_id[sid])
        return out

    def write(
        self,
        key: torch.Tensor,
        content: str,
        weight: Optional[float] = None,
        source: SlotSource = SlotSource.MODEL_INFERRED,
    ) -> MemorySlot:
        """
        写入新槽或更新已有槽（文档 §7.3、§7.4、§11.2 版本化）。
        - 与某槽相似度 >= merge_similarity_threshold：更新该槽（key/content/weight，version+1）。
        - 与已有槽最大相似度 >= write_new_slot_threshold 但 < merge：考虑更新最相似槽（§7.3）。
        - 最大相似度 < write_new_slot_threshold：才写入新槽；满则淘汰。
        """
        key = self._normalize_key(key)
        w = weight if weight is not None else self.config.slot_init_weight
        merge_th = self.config.merge_similarity_threshold
        write_th = self.config.write_new_slot_threshold

        if self._slots:
            best_idx = -1
            best_sim = -1.0
            for i, slot in enumerate(self._slots):
                sim = cosine_similarity(key, slot.key).item()
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            if best_idx >= 0 and best_sim >= write_th:
                slot = self._slots[best_idx]
                old_version = slot.version
                slot.version += 1
                slot.last_accessed_at = time.time()
                slot.weight = max(slot.weight, w)
                if best_sim >= merge_th:
                    slot.key = key.to(slot.key.device)
                    slot.content = content
                    logger.debug(
                        "updated slot(user=%s, id=%s, version=%d->%d) by merge, sim=%.4f",
                        self.user_id,
                        slot.slot_id,
                        old_version,
                        slot.version,
                        best_sim,
                    )
                else:
                    slot.content = content
                    logger.debug(
                        "updated slot(user=%s, id=%s, version=%d->%d) by partial merge, sim=%.4f",
                        self.user_id,
                        slot.slot_id,
                        old_version,
                        slot.version,
                        best_sim,
                    )
                return slot

        if len(self._slots) >= self.config.max_slots_per_user:
            self._evict_one()
        slot = MemorySlot.create(
            user_id=self.user_id,
            key=key,
            content=content,
            weight=w,
            source=source,
        )
        self._slots.append(slot)
        self._slot_by_id[slot.slot_id] = slot
        logger.debug(
            "created new slot(user=%s, id=%s, weight=%.4f, source=%s)",
            self.user_id,
            slot.slot_id,
            slot.weight,
            slot.source.value,
        )
        return slot

    def _evict_one(self) -> None:
        """淘汰一条：按 weight * decay^(now - created_at) 取最小（§7.4）。"""
        if not self._slots:
            return
        now = time.time()
        best_idx = 0
        best_score = float("inf")
        for i, slot in enumerate(self._slots):
            age = now - slot.created_at
            decay = self.config.slot_decay_lambda ** max(age / 3600.0, 0)  # 按小时衰减
            score = slot.weight * decay
            if score < best_score:
                best_score = score
                best_idx = i
        removed = self._slots.pop(best_idx)
        del self._slot_by_id[removed.slot_id]
        if self.graph:
            self.graph.remove_slot(removed.slot_id)
        logger.info(
            "evicted slot(user=%s, id=%s, final_score=%.6f)",
            self.user_id,
            removed.slot_id,
            best_score,
        )

    def apply_decay(self) -> None:
        """对所有槽权重做时间衰减（§11.3）；可定时调用。"""
        now = time.time()
        for slot in self._slots:
            age = now - slot.created_at
            slot.weight *= self.config.slot_decay_lambda ** max(age / 3600.0, 0)

    @property
    def num_slots(self) -> int:
        return len(self._slots)

    @property
    def slot_by_id(self) -> Dict[str, MemorySlot]:
        return self._slot_by_id

    def export_slots_json(self) -> List[Dict[str, Any]]:
        """供 checkpoint 序列化（与 ``clear_and_load_from_json`` 配对）。"""
        return [s.to_dict() for s in self._slots]

    def clear_and_load_from_json(self, rows: List[Dict[str, Any]], *, device: torch.device) -> None:
        """从 ``export_slots_json`` 结果恢复槽列表（覆盖当前内存）。"""
        self._slots.clear()
        self._slot_by_id.clear()
        for row in rows:
            slot = MemorySlot.from_dict(
                row,
                embed_dim=self.config.embed_dim,
                device=device,
                user_id=self.user_id,
            )
            self._slots.append(slot)
            self._slot_by_id[slot.slot_id] = slot
