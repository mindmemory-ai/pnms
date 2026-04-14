# -*- coding: utf-8 -*-
"""
个人神经记忆系统 (PNMS) — PyTorch 实现。

本包提供一个可嵌入到任意上层应用中的「个体化长期记忆引擎」，对应设计文档：docs/pnms.md。
"""

from .versioning import (
    LIBRARY_VERSION,
    MEMORY_FORMAT_VERSION,
    get_library_version,
    get_memory_format_version,
    get_versions,
    peek_checkpoint_versions,
)

__version__ = LIBRARY_VERSION

from .config import PNMSConfig
from .state import PersonalNeuralState
from .slot import MemorySlot, SlotSource
from .memory import MemoryStore
from .graph import MemoryGraph
from .encoder import SimpleQueryEncoder, IdentityEncoder, SentenceEncoder
from .concept import (
    ConceptModule,
    ConceptModuleManager,
    build_augment_training_data,
    form_and_train_one_module,
)
from .context import ContextBuilder
from .client import PNMSClient, HandleQueryResult, ContextResult
from .update import (
    compute_interaction_embedding,
    try_form_concept_modules_from_store,
    update_state,
    update_memory,
)
from .system import PNMS
from .exceptions import (
    ErrorCodes,
    PNMSError,
    ConfigError,
    EncoderError,
    PersistenceError,
    ConceptError,
    GraphError,
    StateError,
    LLMError,
)

def setup_basic_logging(level: str = "INFO") -> None:
    """
    为 PNMS 配置一个简单的 logging 输出，方便在脚本 / 实验环境中快速启用日志。

    生产环境下，推荐由上层应用统一配置 logging；本函数仅做兜底与快捷入口。
    """
    import logging

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

__all__ = [
    "PNMSConfig",
    "PersonalNeuralState",
    "MemorySlot",
    "SlotSource",
    "MemoryStore",
    "MemoryGraph",
    "SimpleQueryEncoder",
    "IdentityEncoder",
    "SentenceEncoder",
    "ConceptModule",
    "ConceptModuleManager",
    "build_augment_training_data",
    "form_and_train_one_module",
    "ContextBuilder",
    "PNMSClient",
    "HandleQueryResult",
    "ContextResult",
    "compute_interaction_embedding",
    "try_form_concept_modules_from_store",
    "update_state",
    "update_memory",
    "PNMS",
    "ErrorCodes",
    "PNMSError",
    "ConfigError",
    "EncoderError",
    "PersistenceError",
    "ConceptError",
    "GraphError",
    "StateError",
    "LLMError",
    "__version__",
    "LIBRARY_VERSION",
    "MEMORY_FORMAT_VERSION",
    "get_library_version",
    "get_memory_format_version",
    "get_versions",
    "peek_checkpoint_versions",
    "setup_basic_logging",
]
