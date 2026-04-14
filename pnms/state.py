# -*- coding: utf-8 -*-
"""
个人神经状态 S_t（文档 §6）。
每个用户维护一个 d 维向量，对长期交互做指数移动平均 (EMA) 摘要。
"""

from __future__ import annotations

import torch
from typing import Optional


class PersonalNeuralState:
    """
    个人神经状态 S_u ∈ R^d（文档 §6.1）。

    更新规则（§6.2）：
        S_{t+1} = (1 - α) * S_t + α * E_t
    其中 E_t 为当前轮交互的嵌入（如 query+response 的聚合）。
    性质：O(d) 更新成本、有界嵌入下状态有界、无需梯度。
    """

    def __init__(self, embed_dim: int, alpha: float = 0.05, device: Optional[torch.device] = None):
        """
        Args:
            embed_dim: 维度 d，与嵌入空间一致。
            alpha: 遗忘因子 α ∈ (0,1)，越大越重视当前轮。
            device: 存放向量的设备；None 则用 CPU。
        """
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.device = device or torch.device("cpu")
        # 初始状态为零向量；也可用随机小量，文档未强制
        self._state: torch.Tensor = torch.zeros(embed_dim, device=self.device, dtype=torch.float32)
        self._initialized = False

    @property
    def state(self) -> torch.Tensor:
        """当前状态向量 S_t，形状 (d,)。只读，更新请用 update()。"""
        return self._state

    def update(self, E_t: torch.Tensor) -> None:
        """
        用当前轮嵌入 E_t 更新状态（文档 §6.2）。

        公式：S_{t+1} = (1 - α) * S_t + α * E_t
        若 E_t 未归一化，这里会先做 L2 归一化再更新，保证数值稳定。

        Args:
            E_t: 当前轮嵌入，形状 (d,) 或 (1, d)；将自动 flatten 并转到正确设备。
        """
        E = E_t.detach().float().to(self.device).flatten()
        if E.shape[0] != self.embed_dim:
            raise ValueError(f"E_t dim {E.shape[0]} != embed_dim {self.embed_dim}")
        # 可选：归一化避免 E 尺度过大导致 S 漂移
        norm = E.norm().clamp(min=1e-8)
        E = E / norm
        if not self._initialized:
            self._state = E.clone()
            self._initialized = True
        else:
            self._state = (1.0 - self.alpha) * self._state + self.alpha * E

    def get_for_retrieval(self) -> torch.Tensor:
        """
        返回用于检索时的用户上下文（文档 §6.4）。
        可与 query 拼接或加权后参与相似度计算，使检索更个性化。
        """
        return self._state

    def reset(self) -> None:
        """重置为零向量并标记未初始化（用于新用户或实验）。"""
        self._state = torch.zeros(self.embed_dim, device=self.device, dtype=torch.float32)
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def load_from_tensor(self, vec: torch.Tensor, *, initialized: bool = True) -> None:
        """从 checkpoint 恢复 S_t（与 ``export`` 配对）。"""
        v = vec.detach().float().to(self.device).flatten()
        if v.shape[0] != self.embed_dim:
            raise ValueError(f"state dim {v.shape[0]} != embed_dim {self.embed_dim}")
        self._state = v.clone()
        self._initialized = bool(initialized)
