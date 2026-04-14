# -*- coding: utf-8 -*-
"""
记忆更新 F(X_t, E_t)（文档 §5.2、§10）。
根据当前轮交互嵌入 E_t 更新：个人神经状态 S、记忆槽 M、记忆图 G；可选触发概念模块 C（§9.2）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

import torch

from .config import PNMSConfig
from .concept import form_and_train_one_module
from .state import PersonalNeuralState
from .memory import MemoryStore
from .graph import MemoryGraph
from .slot import MemorySlot, SlotSource

if TYPE_CHECKING:
    from .concept import ConceptModuleManager


def compute_interaction_embedding(
    query_embed: torch.Tensor,
    response_embed: Optional[torch.Tensor] = None,
    agg: str = "mean",
) -> torch.Tensor:
    """
    由 query（及可选 response）嵌入聚合为当前轮嵌入 E_t（文档 §5.2）。
    agg: "mean" | "query_only"；若 response_embed 为 None 则仅用 query。
    """
    if response_embed is not None and agg == "mean":
        E = (query_embed.flatten() + response_embed.flatten()) / 2.0
    else:
        E = query_embed.flatten()
    return E


def update_state(S: PersonalNeuralState, E_t: torch.Tensor) -> None:
    """更新个人神经状态 S_{t+1} = (1-α)S_t + α·E_t（文档 §6.2）。"""
    S.update(E_t)


def update_memory(
    config: PNMSConfig,
    store: MemoryStore,
    graph: Optional[MemoryGraph],
    query_embed: torch.Tensor,
    response_embed: Optional[torch.Tensor],
    content_to_write: Optional[str],
    activated_slot_ids: List[str],
) -> None:
    """
    单轮记忆更新（文档 §10 记忆更新）。
    - 用 query_embed（及 response_embed）得到 E_t，更新 S（在外部对 S 调用 update_state）；
    - 若 content_to_write 非空，写入新槽或更新已有槽；
    - 对本轮被激活的槽对在图上做共现记录。
    """
    if content_to_write:
        E_t = compute_interaction_embedding(query_embed, response_embed, agg="mean")
        if response_embed is None:
            E_t = query_embed.flatten()
        store.write(
            key=E_t,
            content=content_to_write,
            weight=config.slot_init_weight,
            source=SlotSource.MODEL_INFERRED,
        )
    if graph and len(activated_slot_ids) >= 2:
        graph.record_cooccurrence(activated_slot_ids)


def try_form_concept_modules_from_store(
    config: PNMSConfig,
    store: MemoryStore,
    concept_manager: "ConceptModuleManager",
    state: PersonalNeuralState,
    device: Optional[torch.device] = None,
) -> int:
    """
    当满足 §9.2 形成条件时，对当前槽做 KMeans 聚类并形成概念模块（样本量 N_min、非冷启动期）。
    仅在 slots_and_graph 阶段且槽数 >= concept_min_cluster_size 时执行；需 scikit-learn。
    返回本次新形成的模块数；若已有模块则跳过（避免单次会话内频繁建模块）。
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return 0
    if store.num_slots < config.concept_min_cluster_size:
        return 0
    if getattr(concept_manager, "_modules", None) and len(concept_manager._modules) > 0:
        return 0
    phase = config.in_cold_start_phase(store.num_slots)
    if phase != "slots_and_graph":
        return 0
    slots = list(store.slot_by_id.values())
    if len(slots) < config.concept_min_cluster_size:
        return 0
    keys = torch.stack([s.key.cpu().float().flatten() for s in slots])
    n = keys.shape[0]
    d = keys.shape[1]
    max_concept = getattr(config, "max_concept_modules", 10)
    n_clusters = min(max_concept, max(1, n // config.concept_min_cluster_size))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(keys.numpy())
    state_vec = state.state.cpu().float() if device is None else state.state.to(device).float()
    formed = 0
    for c in range(n_clusters):
        idx = [i for i in range(n) if labels[i] == c]
        if len(idx) < config.concept_min_cluster_size:
            continue
        slot_ids_c = [slots[i].slot_id for i in idx]
        slot_keys_c = [slots[i].key for i in idx]
        center = torch.tensor(km.cluster_centers_[c], dtype=torch.float32)
        if device is not None:
            center = center.to(device)
        form_and_train_one_module(
            concept_manager,
            f"concept_{c}",
            center,
            slot_ids_c,
            slot_keys_c,
            state_vec,
            device=device,
            epochs=20,
            lr=1e-3,
        )
        formed += 1
    return formed
