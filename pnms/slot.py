# -*- coding: utf-8 -*-
"""
记忆槽 m_i = (k_i, v_i, w_i)（文档 §7）。
单条槽的数据结构及来源枚举。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import torch


class SlotSource(Enum):
    """槽内容来源（文档 §13.1）：用户显式声明 vs 模型推断。"""
    USER_DECLARED = "user_declared"
    MODEL_INFERRED = "model_inferred"


@dataclass
class MemorySlot:
    """
    单条记忆槽（文档 §7.1）：m_i = (k_i, v_i, w_i)。

    字段与文档 §13.1 建议一致：
    - slot_id, user_id, key, content, weight, created_at, last_accessed_at, source, version。
    """

    slot_id: str
    user_id: str
    key: torch.Tensor  # (d,) 嵌入键，用于相似度检索
    content: str  # 原始内容，拼入 LLM 上下文
    weight: float
    created_at: float  # 时间戳，用于衰减与淘汰
    last_accessed_at: float
    source: SlotSource = SlotSource.MODEL_INFERRED
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """序列化用：key 转为 list。"""
        return {
            "slot_id": self.slot_id,
            "user_id": self.user_id,
            "key": self.key.cpu().tolist(),
            "content": self.content,
            "weight": self.weight,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "source": self.source.value,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        embed_dim: int,
        device: torch.device,
        user_id: str,
    ) -> "MemorySlot":
        """从 ``to_dict`` 结果恢复槽；``user_id`` 使用当前引擎用户，避免跨用户串数据。"""
        key_list = data.get("key")
        if not isinstance(key_list, list):
            raise ValueError("slot dict missing key tensor list")
        key = torch.tensor(key_list, dtype=torch.float32, device=device).flatten()
        if key.numel() != embed_dim:
            raise ValueError(f"slot key len {key.numel()} != embed_dim {embed_dim}")
        src_raw = data.get("source", SlotSource.MODEL_INFERRED.value)
        try:
            src = SlotSource(str(src_raw))
        except ValueError:
            src = SlotSource.MODEL_INFERRED
        return cls(
            slot_id=str(data["slot_id"]),
            user_id=user_id,
            key=key,
            content=str(data.get("content", "")),
            weight=float(data.get("weight", 1.0)),
            created_at=float(data.get("created_at", 0.0)),
            last_accessed_at=float(data.get("last_accessed_at", 0.0)),
            source=src,
            version=int(data.get("version", 1)),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def create(
        cls,
        user_id: str,
        key: torch.Tensor,
        content: str,
        weight: float = 1.0,
        source: SlotSource = SlotSource.MODEL_INFERRED,
    ) -> "MemorySlot":
        """工厂方法：生成新槽并填时间戳。"""
        now = time.time()
        return cls(
            slot_id=str(uuid.uuid4()),
            user_id=user_id,
            key=key.detach().float(),
            content=content,
            weight=weight,
            created_at=now,
            last_accessed_at=now,
            source=source,
            version=1,
        )

    def bump_access(self, weight_bump: float = 0.05, weight_gamma: Optional[float] = None) -> None:
        """
        被检索命中时调用（文档 §7.4）。
        - 更新 last_accessed_at；
        - 权重：w_i += weight_bump 或 w_i = (1-γ)w_i + γ。
        """
        self.last_accessed_at = time.time()
        if weight_gamma is not None:
            self.weight = (1.0 - weight_gamma) * self.weight + weight_gamma
        else:
            self.weight = self.weight + weight_bump
