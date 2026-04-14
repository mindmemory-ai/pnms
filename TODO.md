# PNMS 库化与生产级优化 TODO

## 1. 库结构与打包
- [进行中] 拆分「库代码」与「示例 / 脚本」，确保 `pnms` 作为纯 Python 包可被外部项目 `pip install -e .` 使用。（当前已基本满足，已有 `pyproject.toml`，后续补充发布流程与元数据）
- [已完成] 增加 `pyproject.toml` 或 `setup.cfg`，声明依赖（`torch`, `sentence-transformers`, `scikit-learn`, `sqlite3` 标准库等）与最低 Python 版本。
- [已完成] 明确对外公开 API（例如 `PNMS`, `PNMSConfig`, 编码器接口、错误类型），其余模块标记为内部实现细节。（已在 `pnms/__init__.py` 中导出核心类与异常，增加 `__version__` 与 `setup_basic_logging`）

## 2. 日志系统
- [部分完成] 引入统一的 `logging` 日志体系，使用命名 logger（如 `pnms.system`、`pnms.graph`、`pnms.memory`、`pnms.concept` 等）。目前已提供 `setup_basic_logging`，`pnms.graph` / `pnms.system` / `pnms.memory` / `pnms.concept` 已接入日志，其它模块后续补齐。
- [部分完成] 为关键操作打点：
  - [部分完成] 初始化与配置解析（包含版本号、重要开关状态）（`pnms.system` 已在初始化时输出关键信息）。
  - [已完成] 槽写入 / 更新 / 淘汰、图边保存 / 加载、概念形成 / 训练 / 保存 / 加载的基础日志。
  - [已完成] 与外部 LLM 调用（仅记录摘要与耗时，不记录隐私内容；`PNMS.handle_query` 中记录 query/context 长度与耗时）。
- [已完成] 提供简单的日志配置帮助函数（例如 `pnms.setup_basic_logging(level="INFO")`），方便集成方快速接入。

## 3. 错误处理与异常模型
- [已完成] 定义库级异常层次结构（例如 `PNMSError` 作为基类，子类包含 `ConfigError`、`EncoderError`、`PersistenceError`、`ConceptError` 等）。（已在 `pnms/exceptions.py` 中实现并在 `__init__.py` 导出）
- [部分完成] 对当前代码中可能抛出原始异常的关键路径（文件读写、SQLite 操作、概念加载、聚类、编码器加载等）加上显式捕获与包装，转化为语义清晰的库异常。（目前主要用于图的保存/加载，其他模块待补）
- [已完成] 在外部 API 层给出稳定的错误语义与文档说明，保证调用方可以有针对性地处理/重试。（README.md 中新增“日志与错误处理”一节，详细说明各异常含义与典型处理方式）

## 4. 对外 API 设计
- [已完成] 设计面向集成方的高层入口类，例如：
  - `PNMS`：作为核心「会话记忆引擎」对象。
  - `PNMSConfig`：可从 dict / JSON / 环境变量初始化，提供基础校验方法（如 `validate()`，并已支持 `to_dict()` / `from_dict()` / `from_env()`）。
  - `PNMSClient` 轻量包装，用于更友好地在 Web 服务中管理多个用户的 PNMS 实例（见 `pnms/client.py`，并在 README 示例中使用）。
- [已完成] 为常见使用场景提供清晰 API：
  - 单轮查询：`PNMS.handle_query(...)` 与 `PNMSClient.handle(...)`。
  - 仅获取上下文（无写入）：`PNMS.get_context_for_query(...)` 与 `PNMSClient.get_context(...)`。
  - 模型 / 概念 / 图的保存与加载：`PNMS.save_concept_modules()`、`PNMS.load_concept_modules()`。
- [部分完成] 统一入参与返回值结构（例如使用 TypedDict 或 dataclass 封装返回的上下文、调试信息），避免到处传裸字符串：
  - [已完成] 新增 `HandleQueryResult` 与 `ContextResult` dataclass，并通过 `PNMSClient` 返回。
  - [未开始] 在更底层 API（如 `PNMS.get_context_for_query`）中返回结构化结果（当前仍返回简单二元组）。

- ## 5. 配置管理与健壮性
- [部分完成] 为 `PNMSConfig` 增加：
  - [已完成] 参数范围校验（如相似度阈值在 [0,1]，衰减间隔为正整数等），提供 `PNMSConfig.validate()` 并在 `PNMS` 初始化时调用。
  - [未开始] 便捷的从/到 JSON、环境变量、命令行参数的转换辅助函数。
- [未开始] 增加对持久化路径的存在性与可写性检查（concept checkpoint 目录、SQLite 文件）。
- [未开始] 在 cold start 阶段、concept formation 触发条件等逻辑上增加保护，防止意外空集 / 维度不匹配导致的崩溃。

## 6. 文档与使用示例
- [已完成] 在仓库根目录新增/完善库级 `README.md`：
  - 概述 PNMS 设计目标与核心组件。
  - 安装方式与依赖说明。
  - 最小可运行示例（含代码片段）。
  - 常见问题（FAQ）与调试建议。
- [未开始] 为示例脚本整理一个清晰目录结构，标明：
  - 基础示例（冷启动 / 槽记忆 / 图扩展）。
  - 进阶示例（概念形成 / 对比有无概念）。
  - 全流程示例（当前的 `full_flow_save_load_verify.py` 等）。
- [未开始] 为对外 API 编写 docstring，后续可自动生成 API 文档。

## 7. 稳定性与测试
- [部分完成] 新增基础单元测试：
  - [已完成] `MemoryStore` 写入 / 更新 / 淘汰基本策略（tests/test_memory_and_graph.py）。
  - [已完成] `MemoryGraph` SQLite 持久化的保存 / 加载一致性（tests/test_memory_and_graph.py）。
  - [已完成] 概念模块的保存 / 加载、基础检索是否正常运行（tests/test_concept_persistence.py）。
  - [已完成] `ContextBuilder` 的 token 预算与补充上下文逻辑（tests/test_context_builder.py）。
- [未开始] 为关键路径增加简单性能回归测试或 benchmark（如写入 N 条后概念形成时间、图扩展对单轮查询的耗时）。
- [未开始] 确保在「不开启概念 / 不开启图 / 只用槽」等不同开关组合下均可正常运行。

## 8. 向后兼容与版本管理
- [已完成] 明确当前版本号（例如 `0.1.0`），写入到库（`pnms/__init__.py` 中的 `__version__`）。
- [已移除] 旧 JSON 图文件兼容与迁移（当前版本起仅支持 SQLite graph.db，不再读取或写入 JSON 图文件）。
- [未开始] 为未来演进预留扩展点（例如 encoder 可插拔接口、外部向量数据库适配层等）。

