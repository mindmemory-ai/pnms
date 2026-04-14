# -*- coding: utf-8 -*-
"""
高层 API：PNMSClient

为上层应用提供一个简单的「多用户记忆引擎」管理器：
- 统一管理 user_id -> PNMS 实例；
- 提供结构化的返回值（包含 response、context、num_slots 等调试信息），避免到处拼裸字符串。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .config import PNMSConfig
from .system import PNMS, LLMReasoner


@dataclass
class HandleQueryResult:
    """单轮对话的结构化结果。"""

    response: str
    context: str
    num_slots_used: int
    phase: str


@dataclass
class ContextResult:
    """只读上下文构建结果。"""

    context: str
    num_slots_used: int
    phase: str


class PNMSClient:
    """
    多用户 PNMS 管理器。

    典型使用方式：
        client = PNMSClient(global_config)
        result = client.handle("user_1", query, llm=my_llm)
    """

    def __init__(self, config: Optional[PNMSConfig] = None) -> None:
        self._base_config = config or PNMSConfig()
        self._engines: Dict[str, PNMS] = {}

    def get_engine(self, user_id: str) -> PNMS:
        """获取或懒初始化某个用户的 PNMS 实例。"""
        if user_id not in self._engines:
            # 为每个用户复制一份配置，避免交叉影响
            cfg = PNMSConfig.from_dict(self._base_config.to_dict())
            self._engines[user_id] = PNMS(config=cfg, user_id=user_id)
        return self._engines[user_id]

    def handle(
        self,
        user_id: str,
        query: str,
        llm: LLMReasoner,
        content_to_remember: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> HandleQueryResult:
        """
        处理某个用户的一轮查询，返回结构化结果（包含上下文与槽使用情况）。
        """
        engine = self.get_engine(user_id)
        # 先构建上下文，便于观测；随后复用相同配置执行完整流程
        ctx, num_slots = engine.get_context_for_query(
            query,
            system_prompt=system_prompt,
            use_concept=None,
        )
        resp = engine.handle_query(
            query=query,
            llm=llm,
            content_to_remember=content_to_remember,
            system_prompt=system_prompt,
        )
        phase = engine.config.in_cold_start_phase(engine.store.num_slots)
        return HandleQueryResult(
            response=resp,
            context=ctx,
            num_slots_used=num_slots,
            phase=phase,
        )

    def get_context(
        self,
        user_id: str,
        query: str,
        system_prompt: Optional[str] = None,
        use_concept: Optional[bool] = None,
    ) -> ContextResult:
        """只读获取某个用户在当前记忆状态下的上下文。"""
        engine = self.get_engine(user_id)
        ctx, num_slots = engine.get_context_for_query(
            query,
            system_prompt=system_prompt,
            use_concept=use_concept,
        )
        phase = engine.config.in_cold_start_phase(engine.store.num_slots)
        return ContextResult(context=ctx, num_slots_used=num_slots, phase=phase)

    def merge(
        self,
        user_id: str,
        source_checkpoint_dir: str,
        source_memory_format_version: Optional[str] = None,
    ) -> dict:
        """
        将指定 checkpoint（目录）中的记忆并入当前用户已加载的运行态记忆。

        详见 ``PNMS.merge_memories``；失败时抛出带 ``code`` 的 ``PNMSError``（见 ``ErrorCodes``）。
        """
        engine = self.get_engine(user_id)
        return engine.merge_memories(
            source_checkpoint_dir=source_checkpoint_dir,
            source_memory_format_version=source_memory_format_version,
        )

