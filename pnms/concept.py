# -*- coding: utf-8 -*-
"""
概念模块（文档 §9）。
轻量可学习模块 f_θ，将 (q, S_u) 映射到概念空间，用于对簇内槽重排（Augment 模式）。
本实现：仅支持 Augment 模式（对比学习训练 f_θ + 投影到 d 维与槽键相似度排序）；不实现 Direct 模式。
支持将已训练的概念神经网络保存到磁盘并从磁盘加载。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .slot import MemorySlot
from .exceptions import ConceptError, PersistenceError
from .versioning import LIBRARY_VERSION, MEMORY_FORMAT_VERSION

logger = logging.getLogger("pnms.concept")


def _cos_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """余弦相似度，支持 (d,) 与 (B, d)；(d,) 与 (1,d) 时用展平点积，避免 1D@2D 形状不合法。"""
    a = a.float()
    b = b.float()
    if a.dim() == 1 and b.dim() == 2:
        b_ = b.squeeze(0)
        return (a @ b_) / (a.norm().clamp(min=1e-8) * b_.norm().clamp(min=1e-8))
    if a.dim() == 2 and b.dim() == 1:
        a_ = a.squeeze(0)
        return (a_ @ b) / (a_.norm().clamp(min=1e-8) * b.norm().clamp(min=1e-8))
    if a.dim() == 1 and b.dim() == 1:
        return (a @ b) / (a.norm().clamp(min=1e-8) * b.norm().clamp(min=1e-8))
    na = a.norm(dim=dim, keepdim=True).clamp(min=1e-8)
    nb = b.norm(dim=dim, keepdim=True).clamp(min=1e-8)
    return (a * b).sum(dim=dim) / (na.squeeze(-1) * nb.squeeze(-1))


class ConceptModule(nn.Module):
    """
    概念模块 f_θ: R^d × R^d -> R^k（文档 §9.3 Augment 模式）。
    输入 (q, S_u)，输出 k 维概念表征；经 proj 投影回 R^d 后与槽键 k_i 做 sim 用于重排。
    """

    def __init__(self, embed_dim: int, concept_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.concept_dim = concept_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, concept_dim),
        )
        # 投影到 embed_dim，使与槽键 k_i 同空间，用于 sim(proj(f_θ(q,S)), k_i) 排序（§9.3）
        self.proj = nn.Linear(concept_dim, embed_dim)

    def forward(self, q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        q: (d,) 或 (B, d); state: (d,) 或 (B, d)。
        输出 (k,) 或 (B, k)。
        """
        if q.dim() == 1:
            q = q.unsqueeze(0)
            state = state.unsqueeze(0)
        x = torch.cat([q, state], dim=-1)
        out = self.mlp(x)
        return out.squeeze(0) if out.size(0) == 1 else out

    def forward_for_ranking(self, q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        返回 R^d 维向量，用于与槽键做相似度排序（§9.6 Augment）。
        """
        h = self.forward(q, state)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.proj(h).squeeze(0)


def build_augment_training_data(
    slot_keys: List[torch.Tensor],
    state: torch.Tensor,
    device: Optional[torch.device] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
    """
    为 Augment 对比学习构建训练样本（文档 §9.5）。
    每条样本：以簇内某槽的键作为 query 的代理，正例为该槽键，负例为簇内其余槽键。
    返回 [(q, state, pos_key, neg_keys), ...]，用于 train_module。
    """
    if len(slot_keys) < 2:
        return []
    state = state.flatten().float()
    if device is not None:
        state = state.to(device)
    out: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]] = []
    keys = [k.flatten().float().to(device) if device else k.flatten().float() for k in slot_keys]
    for i in range(len(keys)):
        q = keys[i]
        pos_key = keys[i]
        neg_keys = [keys[j] for j in range(len(keys)) if j != i]
        out.append((q, state, pos_key, neg_keys))
    return out


def form_and_train_one_module(
    manager: "ConceptModuleManager",
    module_id: str,
    center: torch.Tensor,
    slot_ids: List[str],
    slot_keys: List[torch.Tensor],
    state: torch.Tensor,
    device: Optional[torch.device] = None,
    epochs: int = 20,
    lr: float = 1e-3,
) -> float:
    """
    注册一个概念模块并对其实施 Augment 对比学习训练（文档 §9.4、§9.5）。
    slot_keys 与 slot_ids 顺序一致；训练数据由 build_augment_training_data 从 slot_keys 生成。
    返回该模块最后一轮平均 loss。
    """
    if len(slot_ids) != len(slot_keys) or len(slot_ids) < 2:
        return 0.0
    manager.add_module(module_id, center, slot_ids)
    data = build_augment_training_data(slot_keys, state, device)
    return manager.train_module(module_id, data, epochs=epochs, lr=lr, device=device)


class ConceptModuleManager:
    """
    概念模块管理器（文档 §9.6）。
    维护多个概念模块及其簇中心；按 q 与簇中心相似度选 top-m 模块，再用模块 f_θ 输出（经 proj）与槽键相似度对槽重排（Augment）。
    """

    def __init__(self, embed_dim: int, concept_dim: int = 32, top_m: int = 2):
        self.embed_dim = embed_dim
        self.concept_dim = concept_dim
        self.top_m = top_m
        self._modules: Dict[str, ConceptModule] = {}
        self._centers: Dict[str, torch.Tensor] = {}
        self._cluster_slot_ids: Dict[str, List[str]] = {}

    def add_module(self, module_id: str, center: torch.Tensor, slot_ids: List[str]) -> None:
        """注册一个概念模块及其簇中心与槽 ID 列表。"""
        self._modules[module_id] = ConceptModule(self.embed_dim, self.concept_dim)
        self._centers[module_id] = center.detach().flatten()[: self.embed_dim].float()
        if self._centers[module_id].numel() != self.embed_dim:
            self._centers[module_id] = torch.zeros(self.embed_dim)
        self._cluster_slot_ids[module_id] = list(slot_ids)

    def retrieve_modules(
        self,
        q: torch.Tensor,
        top_m: Optional[int] = None,
    ) -> List[Tuple[str, ConceptModule]]:
        """按 q 与各簇中心相似度取 top-m 个模块（§9.6）。"""
        if not self._centers:
            return []
        m = top_m or self.top_m
        q = q.flatten().float()
        scores = []
        for mid, c in self._centers.items():
            c = c.to(q.device)
            sim = (q @ c) / (q.norm().clamp(min=1e-8) * c.norm().clamp(min=1e-8))
            scores.append((mid, sim.item()))
        scores.sort(key=lambda x: -x[1])
        out = []
        for mid, _ in scores[:m]:
            if mid in self._modules:
                out.append((mid, self._modules[mid]))
        return out

    def train_module(
        self,
        module_id: str,
        training_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]],
        epochs: int = 20,
        lr: float = 1e-3,
        temperature: float = 0.07,
        device: Optional[torch.device] = None,
    ) -> float:
        """
        对指定概念模块做 Augment 对比学习训练（文档 §9.5）。
        training_data: [(q, state, pos_key, neg_keys), ...]，由 build_augment_training_data 生成。
        损失：与该簇内「正确」槽的匹配度，对比学习 -log(exp(sim/τ) / (exp(sim_pos/τ) + Σ exp(sim_neg/τ)))。
        返回最后一轮平均 loss。
        """
        if module_id not in self._modules or not training_data:
            logger.warning("skip training concept %s: no module or empty data", module_id)
            return 0.0
        dev = device or next(self._modules[module_id].parameters()).device
        mod = self._modules[module_id].to(dev).train()
        opt = torch.optim.Adam(mod.parameters(), lr=lr)
        tau = temperature
        last_avg_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0
            for q, state, pos_key, neg_keys in training_data:
                q = q.to(dev).unsqueeze(0)
                state = state.to(dev).unsqueeze(0)
                pos_key = pos_key.to(dev).unsqueeze(0)
                neg_list = [nk.to(dev) for nk in neg_keys]
                concept_vec = mod.forward_for_ranking(q, state)
                sim_pos = _cos_sim(concept_vec, pos_key)
                cv = concept_vec.squeeze(0) if concept_vec.dim() > 1 else concept_vec
                sim_neg = torch.stack([_cos_sim(cv, nk) for nk in neg_list])
                logits = torch.cat([sim_pos.flatten()[:1] / tau, sim_neg / tau]).unsqueeze(0)
                labels = torch.zeros(1, dtype=torch.long, device=dev)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n += 1
            if n > 0:
                last_avg_loss = epoch_loss / n
                logger.debug(
                    "concept %s epoch %d avg_loss=%.6f",
                    module_id,
                    epoch + 1,
                    last_avg_loss,
                )
        mod.eval()
        logger.info(
            "finished training concept %s, last_avg_loss=%.6f (steps=%d)",
            module_id,
            last_avg_loss,
            len(training_data),
        )
        return last_avg_loss

    def augment_slots(
        self,
        slots: List[MemorySlot],
        q: torch.Tensor,
        state: torch.Tensor,
        module_ids: Optional[List[str]] = None,
    ) -> List[MemorySlot]:
        """
        用选中的概念模块对槽重排（§9.6 Augment）。
        使用 f_θ(q,S_u) 经 proj 后的向量与簇内槽键的相似度 sim(proj(f_θ(q,S)), k_i) 排序。
        """
        if not slots or not module_ids:
            return slots
        device = q.device
        q = q.flatten().to(device)
        state = state.flatten().to(device)
        in_cluster: List[MemorySlot] = []
        out_cluster: List[MemorySlot] = []
        for s in slots:
            in_any = False
            for mid in module_ids:
                if mid in self._cluster_slot_ids and s.slot_id in self._cluster_slot_ids[mid]:
                    in_any = True
                    break
            if in_any:
                in_cluster.append(s)
            else:
                out_cluster.append(s)
        if not in_cluster:
            return slots
        mid = module_ids[0]
        mod = self._modules[mid]
        with torch.no_grad():
            concept_vec = mod.forward_for_ranking(q.unsqueeze(0), state.unsqueeze(0))
        concept_vec = concept_vec.flatten()
        scores = []
        for s in in_cluster:
            k = s.key.to(device).flatten().float()
            sim = (concept_vec @ k) / (concept_vec.norm().clamp(min=1e-8) * k.norm().clamp(min=1e-8))
            scores.append((s, sim.item()))
        scores.sort(key=lambda x: -x[1])
        ordered = [x[0] for x in scores]
        return ordered + out_cluster

    def save(self, path: Union[str, Path]) -> None:
        """
        将已训练的概念模块保存到目录（文档 §13.3）。
        目录结构：path/meta.json（embed_dim, concept_dim, top_m, module_ids、格式版本字段）；
        path/{module_id}.pt（state_dict, center, slot_ids）。
        """
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        meta = {
            "embed_dim": self.embed_dim,
            "concept_dim": self.concept_dim,
            "top_m": self.top_m,
            "module_ids": list(self._modules.keys()),
            "memory_format_version": MEMORY_FORMAT_VERSION,
            "pnms_library_version": LIBRARY_VERSION,
        }
        try:
            (root / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            for mid in self._modules:
                mod = self._modules[mid]
                center = self._centers.get(mid)
                slot_ids = self._cluster_slot_ids.get(mid, [])
                payload = {
                    "state_dict": mod.state_dict(),
                    "center": center.cpu() if center is not None else None,
                    "slot_ids": slot_ids,
                }
                torch.save(payload, root / f"{mid}.pt")
            logger.info("saved %d concept modules into %s", len(self._modules), root)
        except Exception as e:
            logger.exception("failed to save concept modules into %s", root)
            raise PersistenceError(f"failed to save concept modules into {root}") from e

    def load(self, path: Union[str, Path], device: Optional[torch.device] = None) -> None:
        """
        从目录加载已保存的概念模块（覆盖当前 _modules/_centers/_cluster_slot_ids）。
        meta.json 中的 embed_dim、concept_dim 需与当前 manager 一致，否则抛出 ValueError。
        """
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"概念模块目录不存在: {root}")
        meta_path = root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"缺少 meta.json: {meta_path}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.exception("failed to read meta.json from %s", meta_path)
            raise PersistenceError(f"failed to read concept meta from {meta_path}") from e
        file_fmt = meta.get("memory_format_version")
        if file_fmt is not None and file_fmt != MEMORY_FORMAT_VERSION:
            logger.warning(
                "checkpoint meta memory_format_version=%s != current %s; "
                "若加载失败请查阅 PNMS 迁移说明",
                file_fmt,
                MEMORY_FORMAT_VERSION,
            )
        embed_dim = meta["embed_dim"]
        concept_dim = meta["concept_dim"]
        top_m = meta.get("top_m", self.top_m)
        if embed_dim != self.embed_dim or concept_dim != self.concept_dim:
            raise ConceptError(
                f"保存的 embed_dim/concept_dim ({embed_dim}/{concept_dim}) 与当前 manager "
                f"({self.embed_dim}/{self.concept_dim}) 不一致，请使用相同维度的 manager 加载"
            )
        self.top_m = top_m
        self._modules.clear()
        self._centers.clear()
        self._cluster_slot_ids.clear()
        dev = device
        loaded = 0
        for mid in meta["module_ids"]:
            pt_path = root / f"{mid}.pt"
            if not pt_path.exists():
                continue
            try:
                try:
                    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
                except TypeError:
                    payload = torch.load(pt_path, map_location="cpu")
                mod = ConceptModule(embed_dim, concept_dim)
                mod.load_state_dict(payload["state_dict"], strict=True)
                if dev is not None:
                    mod = mod.to(dev)
                self._modules[mid] = mod
                if payload.get("center") is not None:
                    c = payload["center"].float()
                    if dev is not None:
                        c = c.to(dev)
                    self._centers[mid] = c.flatten()[:embed_dim]
                else:
                    self._centers[mid] = torch.zeros(embed_dim, device=dev)
                self._cluster_slot_ids[mid] = list(payload.get("slot_ids", []))
                loaded += 1
            except Exception as e:
                logger.exception("failed to load concept module %s from %s", mid, pt_path)
                raise PersistenceError(f"failed to load concept module {mid} from {pt_path}") from e
        logger.info("loaded %d concept modules from %s", loaded, root)
