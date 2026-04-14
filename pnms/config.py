# -*- coding: utf-8 -*-
"""
PNMS 全局配置。
对应文档 docs/pnms.md 中各处超参数与阈值，便于实验时统一调整。
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional, Type, TypeVar
import os

from .exceptions import ConfigError

# 上下文补足：槽不足时由外部 RAG/知识库提供额外内容（§12.3）
# 签名 (query: str, num_current_slots: int, max_slots: int) -> str，返回拼入上下文的补充文本
ContextSupplementCallable = Callable[[str, int, int], str]


T = TypeVar("T", bound="PNMSConfig")


@dataclass
class PNMSConfig:
    """
    个人神经记忆系统 (PNMS) 的完整配置。

    设计说明（见文档 §4 整体架构、§12 冷启动、§11 一致性与遗忘）：
    - 嵌入与状态维度需一致，保证 S_t 与槽键在同一空间。
    - 冷启动阈值 N0、N1 控制何时开启槽检索、图扩展、概念模块。
    """

    # ----- 嵌入与维度（文档 §6 个人神经状态、§7 记忆槽） -----
    embed_dim: int = 64
    """嵌入维度 d；与文档中 S_u ∈ R^d、k_i ∈ R^d 一致。使用 SentenceEncoder 时会被自动同步为模型维度（如 384）。"""

    # ----- 个人神经状态 S_t（文档 §6.2 更新规则） -----
    state_alpha: float = 0.05
    """EMA 遗忘因子 α：S_{t+1} = (1-α)S_t + α·E_t。越小历史权重越大，越大则越重视当前轮。"""

    # ----- 记忆槽（文档 §7） -----
    max_slots_per_user: int = 1000
    """每用户最大槽数，超出时触发淘汰（§7.5）。"""
    slot_init_weight: float = 1.0
    """新写入槽的初始重要性权重 w_i（§7.4）。"""
    slot_weight_bump_on_access: float = 0.05
    """槽被检索命中时权重的增加量 β（§7.4 常用即重要）。"""
    slot_decay_lambda: float = 0.995
    """槽权重时间衰减系数 λ：w_i *= λ^(t-t_i)（§11.3 遗忘）。"""
    retrieval_top_k: int = 5
    """检索时返回的 top-k 槽数（§7.2、§10.2）。"""
    min_slots_to_use_memory: int = 1
    """检索到的槽数少于此值时可不拼入上下文，视为“无记忆”（§12.3）。"""

    # ----- 记忆图（文档 §8） -----
    graph_enabled: bool = True
    """是否启用图扩展（冷启动阶段可关闭，§12.2）。"""
    graph_cooccur_delta: float = 0.1
    """共现时边权增加量 δ：w_ij^{t+1} = w_ij^t + δ（§8.2）。"""
    graph_edge_decay: float = 0.99
    """边权全局衰减 λ（§8.2）。"""
    graph_max_neighbors_per_slot: int = 3
    """图扩展时每槽最多取几个邻居（§8.4）。"""
    graph_max_expanded_total: int = 15
    """图扩展后总槽数上限（去重后截断）。"""

    # ----- 冷启动（文档 §12.2 分阶段策略） -----
    cold_start_n0: int = 10
    """槽数 < N0 时：仅更新 S 与写槽，不检索、不图、不概念（纯 LLM 阶段）。"""
    cold_start_n1: int = 100
    """槽数 < N1 时：开启槽检索与图边更新，但不建概念模块；>= N1 才考虑图扩展与概念。"""

    # ----- 概念模块（文档 §9） -----
    concept_enabled: bool = True
    """是否启用概念模块（冷启动或槽数少时可关闭）。"""
    concept_min_cluster_size: int = 20
    """形成概念模块的最小簇大小 N_min（§9.2）。"""
    concept_dim: int = 32
    """概念表征输出维度 k（§9.3 Augment 模式）。"""
    concept_top_m: int = 2
    """检索时最多选用几个概念模块（§9.6）。"""
    max_concept_modules: int = 10
    """每用户最多概念模块数 M（§9.4）；KMeans 的 K 上界。"""
    concept_formation_interval_rounds: Optional[int] = 50
    """每 N 轮尝试一次概念形成（§9.2 周期性重聚类）；None 表示不自动形成。"""
    concept_checkpoint_dir: Optional[str] = None
    """概念模块持久化目录（§13.3）。若设置：启动时从该目录加载已训练模块，形成/训练后自动保存。可按用户设为不同子目录（如 concepts/user_1）。"""

    # ----- 上下文构建（文档 §10.3） -----
    max_context_slots: int = 10
    """拼入上下文的记忆条数上限，超出则按相似度/权重截断。"""
    max_context_tokens_estimate: Optional[int] = None
    """最大记忆 token 数 L_max（§10.3）；超出则从尾部截断。None 表示仅按条数截断。"""

    # ----- 记忆更新与写入（文档 §10、§11） -----
    write_new_slot_threshold: float = 0.85
    """新事件与已有槽最大相似度低于此阈值时才写入新槽，否则考虑更新已有槽（§7.3）。需小于 merge_similarity_threshold；示例中若需快速攒槽可适当调低。"""
    merge_similarity_threshold: float = 0.9
    """与某槽相似度高于此则更新该槽而非新增（§7.3）。"""
    slot_access_weight_gamma: float = 0.1
    """命中时权重更新：w_i <- (1-γ)w_i + γ（§7.4）。"""

    # ----- 遗忘与衰减触发（文档 §11.4） -----
    decay_every_n_rounds: Optional[int] = 10
    """每 N 轮触发一次槽权重衰减与图边衰减（§11.3、§11.4）；None 表示不自动触发。"""

    # ----- 冷启动上下文补足（文档 §12.3） -----
    context_supplement: Optional[ContextSupplementCallable] = None
    """槽数不足时可选调用：用全局 RAG/知识库补足上下文；(query, num_slots, max_slots)->str。"""

    # ----- 配置辅助方法 -----

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为可 JSON 序列化的 dict（去掉不可序列化的回调）。
        便于持久化与从外部配置源加载。
        """
        data = asdict(self)
        # 回调类字段不宜直接序列化，交由上层以代码或注册表方式注入
        data.pop("context_supplement", None)
        return data

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        从 dict 创建配置实例。未知字段会被忽略，缺失字段使用默认值。
        """
        allowed_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in allowed_keys}
        return cls(**filtered)  # type: ignore[arg-type]

    @classmethod
    def from_env(cls: Type[T], prefix: str = "PNMS_") -> T:
        """
        从环境变量构造配置。

        环境变量名格式：PNMS_FIELD_NAME（按字段名大写，点号改为下划线）。
        仅支持基础类型：int/float/bool/str。无法解析的值会被忽略。
        """
        data: Dict[str, Any] = {}
        for field_name, field_def in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
            env_key = prefix + field_name.upper()
            if env_key not in os.environ:
                continue
            raw = os.environ[env_key]
            target_type = field_def.type
            value: Any = raw
            # 简单的类型推断与转换
            try:
                if target_type in (int, Optional[int]):
                    value = int(raw)
                elif target_type in (float, Optional[float]):
                    value = float(raw)
                elif target_type in (bool, Optional[bool]):
                    value = raw.lower() in ("1", "true", "yes", "y", "on")
                else:
                    # 其余类型按字符串处理，由 from_dict 决定是否接受
                    value = raw
            except ValueError:
                # 若转换失败，跳过该字段
                continue
            data[field_name] = value
        return cls.from_dict(data)

    def update_from_env(self, prefix: str = "PNMS_") -> None:
        """
        使用环境变量更新当前配置实例，仅覆盖在环境中显式设置的字段。
        """
        env_cfg = self.from_env(prefix=prefix)
        for k, v in env_cfg.to_dict().items():
            setattr(self, k, v)

    def validate(self) -> None:
        """
        对关键超参数做范围与逻辑校验，便于在应用启动阶段尽早发现配置错误。

        若配置不合法，将抛出 ConfigError。
        """
        if self.embed_dim <= 0:
            raise ConfigError(f"embed_dim must be positive, got {self.embed_dim}")
        if not (0.0 < self.state_alpha <= 1.0):
            raise ConfigError(f"state_alpha must be in (0,1], got {self.state_alpha}")
        if self.max_slots_per_user <= 0:
            raise ConfigError(f"max_slots_per_user must be positive, got {self.max_slots_per_user}")
        if not (0.0 <= self.slot_weight_bump_on_access <= 1.0):
            raise ConfigError(
                f"slot_weight_bump_on_access must be in [0,1], got {self.slot_weight_bump_on_access}"
            )
        if not (0.0 < self.slot_decay_lambda <= 1.0):
            raise ConfigError(f"slot_decay_lambda must be in (0,1], got {self.slot_decay_lambda}")
        if self.retrieval_top_k <= 0:
            raise ConfigError(f"retrieval_top_k must be positive, got {self.retrieval_top_k}")
        if self.graph_cooccur_delta < 0.0:
            raise ConfigError(f"graph_cooccur_delta must be non-negative, got {self.graph_cooccur_delta}")
        if not (0.0 < self.graph_edge_decay <= 1.0):
            raise ConfigError(f"graph_edge_decay must be in (0,1], got {self.graph_edge_decay}")
        if self.graph_max_neighbors_per_slot < 0:
            raise ConfigError(
                f"graph_max_neighbors_per_slot must be non-negative, got {self.graph_max_neighbors_per_slot}"
            )
        if self.graph_max_expanded_total <= 0:
            raise ConfigError(f"graph_max_expanded_total must be positive, got {self.graph_max_expanded_total}")
        if self.cold_start_n0 < 0 or self.cold_start_n1 < 0:
            raise ConfigError(f"cold_start_n0/n1 must be non-negative, got {self.cold_start_n0},{self.cold_start_n1}")
        if self.cold_start_n1 < self.cold_start_n0:
            raise ConfigError(
                f"cold_start_n1 ({self.cold_start_n1}) must be >= cold_start_n0 ({self.cold_start_n0})"
            )
        if self.concept_min_cluster_size <= 0:
            raise ConfigError(
                f"concept_min_cluster_size must be positive, got {self.concept_min_cluster_size}"
            )
        if self.concept_dim <= 0:
            raise ConfigError(f"concept_dim must be positive, got {self.concept_dim}")
        if self.concept_top_m <= 0:
            raise ConfigError(f"concept_top_m must be positive, got {self.concept_top_m}")
        if self.max_concept_modules <= 0:
            raise ConfigError(f"max_concept_modules must be positive, got {self.max_concept_modules}")
        if self.write_new_slot_threshold < 0.0 or self.merge_similarity_threshold < 0.0:
            raise ConfigError(
                f"similarity thresholds must be non-negative, got write_new_slot_threshold="
                f"{self.write_new_slot_threshold}, merge_similarity_threshold={self.merge_similarity_threshold}"
            )
        if self.write_new_slot_threshold > self.merge_similarity_threshold:
            raise ConfigError(
                "write_new_slot_threshold must be <= merge_similarity_threshold, "
                f"got {self.write_new_slot_threshold} > {self.merge_similarity_threshold}"
            )
        if self.decay_every_n_rounds is not None and self.decay_every_n_rounds <= 0:
            raise ConfigError(
                f"decay_every_n_rounds must be positive when set, got {self.decay_every_n_rounds}"
            )

    def in_cold_start_phase(self, num_slots: int) -> str:
        """
        返回当前冷启动阶段标识（文档 §12.2）。
        - "pure_llm": 槽数 < N0
        - "slots_only": N0 <= 槽数 < N1
        - "slots_and_graph": 槽数 >= N1
        """
        if num_slots < self.cold_start_n0:
            return "pure_llm"
        if num_slots < self.cold_start_n1:
            return "slots_only"
        return "slots_and_graph"
