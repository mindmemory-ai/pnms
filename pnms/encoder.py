# -*- coding: utf-8 -*-
"""
查询编码器（文档 §4、§13.5）。
将原始 query（及可选 response）编码为向量 q ∈ R^d，与槽键同一空间。
- SimpleQueryEncoder：字符级简单编码，仅用于无依赖实验。
- SentenceEncoder：基于 sentence-transformers 的语义编码，检索质量与“记忆感”更好。
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn


class SimpleQueryEncoder(nn.Module):
    """
    简单查询编码器：将 token ids 或已有向量映射到 embed_dim 维。

    若输入已是向量（如外部 API 返回的 embedding），则过一层线性层对齐维度；
    若输入为 token ids（LongTensor），则先 embedding 再池化再线性。
    实验阶段可用此模块；生产环境建议替换为 sentence-transformers 等。
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int = 10000,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self._use_embed = True

    def forward(
        self,
        x: Union[torch.Tensor, List[int], List[List[int]]],
    ) -> torch.Tensor:
        """
        编码为 (d,) 或 (batch, d)。单条 query 时输出 (d,)。

        x: 若为 LongTensor 形状 (seq_len,) 或 (batch, seq_len)，则走 embedding+mean；
           若为 FloatTensor 形状 (d_in,) 或 (batch, d_in)，则仅做线性投影（需 d_in==embed_dim 或额外一层）。
        """
        if isinstance(x, (list,)):
            x = torch.tensor(x, dtype=torch.long)
        if x.dtype == torch.long:
            # (L,) or (B, L) -> emb (B, L, embed_dim) 或 (L, embed_dim)
            emb = self.emb(x)
            if emb.dim() == 2:
                out = emb.mean(dim=0)  # (L, d) -> (d,)
            else:
                out = emb.mean(dim=1)  # (B, L, d) -> (B, d)
            return self.proj(out).squeeze(0) if out.dim() == 2 and out.size(0) == 1 else self.proj(out)
        else:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.proj(x).squeeze(0) if x.size(0) == 1 else self.proj(x)


class IdentityEncoder(nn.Module):
    """
    恒等编码器：输入已是 embed_dim 维向量时直接返回，仅做 L2 归一化。
    用于：已有外部嵌入（如 OpenAI embedding）时，只保证归一化与设备一致。
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.flatten()
        else:
            x = x.view(x.size(0), -1)
        if x.size(-1) != self.embed_dim:
            raise ValueError(f"Expected dim {self.embed_dim}, got {x.size(-1)}")
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (x / norm).squeeze(0) if x.dim() == 2 and x.size(0) == 1 else (x / norm)


# 默认 sentence embedding 模型：多语言含中文，384 维，体积较小
DEFAULT_SENTENCE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SentenceEncoder(nn.Module):
    """
    基于 sentence-transformers 的语义编码器。
    输入为原始文本，输出为 L2 归一化的向量，与槽键同一语义空间，检索命中率与“记忆感”优于字符级编码。
    需安装：pip install sentence-transformers
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SENTENCE_MODEL,
        device: Optional[Union[str, torch.device]] = None,
        normalize: bool = True,
    ):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("请安装 sentence-transformers：pip install sentence-transformers")
        self._device = device or "cpu"
        self._model = SentenceTransformer(model_name, device=str(self._device))
        self.embed_dim = self._model.get_sentence_embedding_dimension()
        self.normalize = normalize

    def encode_text(self, text: str) -> torch.Tensor:
        """将单条文本编码为 (embed_dim,) 的 tensor，已 L2 归一化（若 normalize=True）。"""
        if not text or not text.strip():
            text = " "
        emb = self._model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            device=str(self._device),
        )
        if emb.dim() == 2:
            emb = emb.squeeze(0)
        return emb

    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        编码一条或多条文本。
        - 单条 str：返回 (embed_dim,)
        - List[str]：返回 (N, embed_dim)
        """
        if isinstance(texts, str):
            return self.encode_text(texts)
        if not texts:
            return self.encode_text(" ")
        out = self._model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            device=str(self._device),
        )
        return out
