# -*- coding: utf-8 -*-
"""
PNMS 异常类型定义。

对外只暴露语义化的异常类，避免调用方直接依赖底层实现细节（如 sqlite3、sklearn 等）。
"""

from __future__ import annotations

from typing import Optional, Type, TypeVar


class ErrorCodes:
    """对外稳定错误码（字符串），便于调用方按码分支处理。"""

    MERGE_NOT_IMPLEMENTED = "E_MERGE_NOT_IMPLEMENTED"
    MERGE_VERSION_INCOMPATIBLE = "E_MERGE_VERSION_INCOMPATIBLE"
    MERGE_INVALID_ARGUMENT = "E_MERGE_INVALID_ARGUMENT"
    MERGE_CHECKPOINT_NOT_FOUND = "E_MERGE_CHECKPOINT_NOT_FOUND"


class PNMSError(Exception):
    """
    PNMS 库的基类异常。

    支持可选的 `cause` 字段，便于在日志中保留底层异常信息，又不向上层泄露具体实现细节。
    """

    def __init__(
        self,
        message: str = "",
        *,
        cause: Optional[BaseException] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.cause = cause
        self.code = code

    def __str__(self) -> str:  # pragma: no cover - 简单覆盖字符串表示
        base = super().__str__()
        if self.code:
            base = f"[{self.code}] {base}"
        if self.cause is not None:
            return f"{base} (cause={self.cause!r})"
        return base


class ConfigError(PNMSError):
    """配置不合法或不一致。"""


class EncoderError(PNMSError):
    """编码器加载或前向过程中的错误。"""


class PersistenceError(PNMSError):
    """持久化（保存 / 加载概念、图、状态）相关错误。"""


class ConceptError(PNMSError):
    """概念模块构建、训练或推理阶段的错误。"""


class GraphError(PNMSError):
    """图结构构建或查询中的错误。"""


class StateError(PNMSError):
    """
    个人神经状态（如 S_t 的维度、更新、序列化）相关错误。

    示例：状态向量维度与配置不一致，或从持久化状态恢复失败。
    """


class LLMError(PNMSError):
    """
    与上层 LLM 调用相关的错误。

    PNMS 本身不直接依赖具体 LLM 实现；当在集成层需要区分
    「记忆子系统正常，但 LLM 调用失败」时，可抛出该异常。
    """


TExc = TypeVar("TExc", bound=PNMSError)


def wrap_error(exc_type: Type[TExc], message: str, cause: BaseException) -> TExc:
    """
    辅助函数：用语义化异常类型包装底层异常。

    用法示例：

    try:
        ...
    except sqlite3.Error as e:
        raise wrap_error(PersistenceError, "failed to save graph", e)
    """

    return exc_type(message, cause=cause)


