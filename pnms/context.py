# -*- coding: utf-8 -*-
"""
上下文构建（文档 §10.3、§13.5）。
将检索到的记忆槽、用户状态摘要等组织成最终 prompt 字符串，供 LLM 使用。
§10.3：槽按相似度/权重排序，长度预算 L_max 超出则从尾部截断。
"""

from __future__ import annotations

from typing import Callable, List, Optional

from .slot import MemorySlot


def _default_token_estimate(text: str) -> int:
    """无 tokenizer 时用字符数/4 粗估 token 数。"""
    return max(1, len(text) // 4)


class ContextBuilder:
    """
    上下文构建器（文档 §10.3）。
    槽按相似度/权重排序后截断到 max_context_slots；若设 L_max 则按 token 数截断。
    格式化为「记忆：content」。
    """

    def __init__(
        self,
        max_context_slots: int = 10,
        max_context_tokens: Optional[int] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        slot_prefix: str = "记忆",
    ):
        self.max_context_slots = max_context_slots
        self.max_context_tokens = max_context_tokens
        self.token_counter = token_counter or _default_token_estimate
        self.slot_prefix = slot_prefix

    def build(
        self,
        slots: List[MemorySlot],
        state_summary: Optional[str] = None,
        system_prompt: Optional[str] = None,
        context_supplement: Optional[str] = None,
    ) -> str:
        """
        构建完整上下文字符串（§10.3、§12.3）。
        记忆部分先按条数截断到 max_context_slots，若设 max_context_tokens 则再按 token 数从尾部截断。
        context_supplement：槽不足时由外部 RAG/知识库提供的补充文本（§12.3）。
        """
        parts = []
        if system_prompt:
            parts.append(system_prompt.strip())
        if state_summary:
            parts.append(f"[用户摘要] {state_summary.strip()}")
        if context_supplement:
            parts.append(f"[补充上下文]\n{context_supplement.strip()}")
        if slots:
            memory_lines = []
            budget = self.max_context_tokens
            for s in slots[: self.max_context_slots]:
                line = f"{self.slot_prefix}：{s.content}"
                if budget is not None:
                    need = self.token_counter(line)
                    if need > budget:
                        break
                    budget -= need
                memory_lines.append(line)
            parts.append("\n".join(memory_lines))
        return "\n\n".join(parts) if parts else ""

    def build_memory_only(self, slots: List[MemorySlot]) -> str:
        """仅返回记忆部分，便于与外部 prompt 拼接；同样受 max_context_slots 与 max_context_tokens 约束。"""
        if not slots:
            return ""
        lines = []
        budget = self.max_context_tokens
        for s in slots[: self.max_context_slots]:
            line = f"{self.slot_prefix}：{s.content}"
            if budget is not None:
                need = self.token_counter(line)
                if need > budget:
                    break
                budget -= need
            lines.append(line)
        return "\n".join(lines)
