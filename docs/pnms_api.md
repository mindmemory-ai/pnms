# PNMS 对外接口说明

本文件面向**调用方 / 上层服务**，总结 PNMS 库的主要类与函数，便于集成到 `mindmemory` / `openclaw-mmem` 等项目中。

---

## 1. 安装与导入

```bash
cd pnms
pip install -r requirements.txt
pip install -e .
```

在代码中典型导入方式：

```python
from pnms import (
    PNMS,
    PNMSClient,
    PNMSConfig,
    SimpleQueryEncoder,
    SentenceEncoder,
    HandleQueryResult,
    ContextResult,
    PNMSError,
    ErrorCodes,
    ConfigError,
    PersistenceError,
    LLMError,
    setup_basic_logging,
)
```

### 1.1 程序版本与记忆文件格式版本

PNMS 将两类版本**分开维护**，便于集成方判断「只换库」还是「可能要迁移 checkpoint」：

| 概念 | 含义 | 典型变更 |
|------|------|----------|
| **库程序版本** | 当前安装的 Python 包版本（与 `pnms.__version__` 一致） | 修 bug、优化、API 兼容扩展 |
| **记忆文件格式版本** | 磁盘 checkpoint 的布局与字段语义（`meta.json`、`graph.db`、`*.pt`） | 增删 meta 字段、改 SQLite 表结构、改 `.pt` payload |

二者可能不同步：例如库已升级到 `0.2.0`，但记忆格式仍为 `1.0.0`，此时调用方通常只需 `pip install -U pnms`。若**记忆格式版本**升级（尤其主版本号变化），集成方需查阅变更说明，必要时迁移旧数据或调整读写逻辑。

查询 API（推荐在启动或加载 checkpoint 前调用）：

```python
from pnms import (
    __version__,
    LIBRARY_VERSION,
    MEMORY_FORMAT_VERSION,
    get_library_version,
    get_memory_format_version,
    get_versions,
    peek_checkpoint_versions,
)

assert __version__ == LIBRARY_VERSION == get_library_version()
assert get_memory_format_version() == MEMORY_FORMAT_VERSION
print(get_versions())  # {"library": "...", "memory_format": "..."}

# 不加载完整模型，仅查看某 checkpoint 目录里记录的格式版本（用于兼容性预判）
info = peek_checkpoint_versions("data/pnms_ckpt")
# memory_format_in_meta / memory_format_in_graph：磁盘上写入的版本；缺省为 None（旧 checkpoint）
# current_library / current_memory_format：当前进程中的库与格式版本
```

保存概念与图时，库会向 `meta.json` 写入 `memory_format_version`、`pnms_library_version`（审计用），向 `graph.db` 的 `pnms_meta` 表写入 `memory_format_version`。加载时若磁盘版本与当前 `MEMORY_FORMAT_VERSION` 不一致，会记录 **warning**，不自动迁移；重大不兼容时请以文档与异常说明为准。

---

## 2. 配置：`PNMSConfig`

### 2.1 创建与校验

```python
from pnms import PNMSConfig

config = PNMSConfig(
    embed_dim=64,
    concept_enabled=True,
    graph_enabled=True,
    concept_checkpoint_dir="data/pnms_ckpt",
)

# 启动时做一次参数合法性校验
config.validate()
```

若配置不合法（如阈值越界、冷启动 N0/N1 矛盾），会抛出 `ConfigError`。

### 2.2 从 dict / JSON 创建

```python
raw = {
    "embed_dim": 128,
    "cold_start_n0": 5,
    "cold_start_n1": 50,
}
config = PNMSConfig.from_dict(raw)
```

`to_dict()` 可用于序列化配置（会自动去掉回调等不可序列化字段）：

```python
cfg_dict = config.to_dict()
```

### 2.3 从环境变量覆盖：`from_env` / `update_from_env`

环境变量命名规则：`PNMS_字段名大写`，例如：

```bash
export PNMS_EMBED_DIM=384
export PNMS_GRAPH_ENABLED=false
```

两种用法：

```python
# 1. 直接从环境变量构造新配置
config = PNMSConfig.from_env(prefix="PNMS_")

# 2. 在已有配置上增量覆盖
config = PNMSConfig()
config.update_from_env(prefix="PNMS_")
```

---

## 3. 核心引擎：`PNMS`

### 3.1 初始化

```python
import torch
from pnms import PNMS, PNMSConfig, SimpleQueryEncoder

device = torch.device("cpu")
config = PNMSConfig(concept_enabled=True, graph_enabled=True)
encoder = SimpleQueryEncoder(embed_dim=config.embed_dim, vocab_size=10000)

engine = PNMS(
    config=config,
    user_id="user_1",
    encoder=encoder,
    device=device,
)
```

`PNMS` 内部会：

- 校验配置（`config.validate()`）  
- 初始化个人神经状态、记忆槽存储、记忆图、概念管理器、上下文构建器  
- 若 `concept_checkpoint_dir` 存在，会尝试加载概念模块与 `graph.db`

### 3.2 处理单轮查询：`handle_query`

```python
def my_llm(query: str, context: str) -> str:
    # 上层自己调用 Ollama / OpenAI / 其他模型
    return "示例回复"

answer = engine.handle_query(
    query="我喜欢用什么语言写算法？",
    llm=my_llm,  # 签名: (query, context_str) -> response_str
    content_to_remember="用户偏好：使用 Python 写算法。",
    system_prompt="你是个人助手，请严格依据记忆回答。",
)
```

行为概要：

1. 对 `query` 做编码得到向量 `q`。  
2. 根据当前槽数决定冷启动阶段（pure_llm / slots_only / slots_and_graph）。  
3. 进行槽检索、可选图扩展、可选概念 Augment。  
4. 用 `ContextBuilder` 生成上下文字符串 `context_str`。  
5. 若 `response` 形参为空，则调用 `llm(query, context_str)`；调用失败会抛出 `LLMError`。  
6. 用 `query` + `response` 更新神经状态、写槽、更新图边、按配置周期性尝试概念形成与持久化。

### 3.3 只读获取上下文：`get_context_for_query`

```python
context, num_slots = engine.get_context_for_query(
    query="二次方程求根公式？",
    system_prompt="你是个人助手。",
    use_concept=True,  # None: 按配置决定；True/False: 强制（有模块时）
)
```

特点：

- 不调用 LLM、不更新状态、不写槽，仅用于观察“当前记忆会给 LLM 什么提示词”。  
- 返回值：`(context_str, num_slots_used)`。

### 3.4 概念与图持久化：`save_concept_modules` / `load_concept_modules`

```python
# 保存到 config.concept_checkpoint_dir 或传入的自定义目录
engine.save_concept_modules()

# 从目录加载已保存的概念与图（graph.db）
engine.load_concept_modules()

# 指定调用方可接受的最大记忆格式版本（默认当前 PNMS 的 MEMORY_FORMAT_VERSION）
# 仅保证向下兼容：checkpoint_version <= expected_memory_format_version
engine.load_concept_modules(expected_memory_format_version="1.0.0")
```

持久化内容：

- `meta.json` + `{module_id}.pt`：概念模块元数据与权重  
- `graph.db`：记忆图边表（SQLite 单文件）

加载前版本校验规则：

- 默认 `expected_memory_format_version=MEMORY_FORMAT_VERSION`（当前运行中的 PNMS 支持版本）。  
- 支持调用方显式传入 `expected_memory_format_version` 做更严格约束。  
- PNMS 仅保证**向下兼容**，不保证向上兼容：若 checkpoint 版本高于 `expected_memory_format_version`，会抛出 `PersistenceError`，调用方可据此中止流程并提示升级。

---

## 4. 多用户管理：`PNMSClient`

当上层服务需要同时管理多个 `user_id` 时，推荐使用 `PNMSClient`。

### 4.1 初始化

```python
from pnms import PNMSClient, PNMSConfig

config = PNMSConfig(concept_enabled=True, graph_enabled=True)
client = PNMSClient(config)
```

`PNMSClient` 内部会按需懒加载 `PNMS` 实例，并按用户隔离。

### 4.2 处理单轮查询：`PNMSClient.handle`

```python
from pnms import HandleQueryResult

def my_llm(query: str, context: str) -> str:
    return "示例回复"

result: HandleQueryResult = client.handle(
    user_id="user_1",
    query="我喜欢用什么语言写算法？",
    llm=my_llm,
    content_to_remember="用户偏好：使用 Python 写算法。",
    system_prompt="你是个人助手，请严格依据记忆回答。",
)

print(result.response)
print(result.context)
print(result.num_slots_used, result.phase)
```

`HandleQueryResult` 字段：

- `response: str`：最终回复文本。  
- `context: str`：本轮发送给 LLM 的上下文字符串（含系统 prompt 与记忆）。  
- `num_slots_used: int`：本轮上下文中使用的记忆槽数量。  
- `phase: str`：当前冷启动阶段（`"pure_llm"` / `"slots_only"` / `"slots_and_graph"`）。

### 4.3 只读上下文：`PNMSClient.get_context`

```python
from pnms import ContextResult

ctx_result: ContextResult = client.get_context(
    user_id="user_1",
    query="二次方程求根公式？",
    system_prompt="你是个人助手。",
    use_concept=True,
)

print(ctx_result.context)
print(ctx_result.num_slots_used, ctx_result.phase)
```

`ContextResult` 字段：

- `context: str`：构建好的上下文字符串。  
- `num_slots_used: int`：使用的槽数量。  
- `phase: str`：当前冷启动阶段。

### 4.4 合并指定记忆 checkpoint：`merge_memories` / `PNMSClient.merge`

当调用方希望把「外部 PNMS checkpoint 目录」并入**当前已在内存中加载**的用户记忆时，使用以下 API（例如 `mindmemory-client` 在解密 bundle 到临时目录后调用）：

```python
# 低层：直接在 PNMS 上调用
engine.merge_memories(
    source_checkpoint_dir="/path/to/another_ckpt",
    source_memory_format_version=None,  # None 表示自动探测 meta.json / graph.db
)

# 高层：通过 PNMSClient 调用（推荐多用户服务）
client.merge(
    user_id="user_1",
    source_checkpoint_dir="/path/to/another_ckpt",
    source_memory_format_version=None,
)
```

语义概要：

- **当前简化实现**：只读取源目录 ``memory_slots.json``，逐条按与在线更新相同的策略写入当前 ``MemoryStore``（相似度合并 / 新槽）。  
- **暂不处理**：``graph.db``、概念模块（``meta.json`` + ``*.pt``）、``memory_session.pt``。  
- **返回值**：返回一个统计字典，包含 ``source_checkpoint_dir``、``source_memory_format_version``、``merged_slots``、``before_slots``、``after_slots``，便于调用方直接拿到结果。

版本约束：

- 仅允许 **source_memory_format_version <= 当前运行 MEMORY_FORMAT_VERSION**（仅向下兼容）。若源更新，则报错 ``ErrorCodes.MERGE_VERSION_INCOMPATIBLE``。

---

## 5. 编码器：`SimpleQueryEncoder` / `SentenceEncoder`

### 5.1 SimpleQueryEncoder

字符级简单编码器，适合快速测试与无外部依赖环境：

```python
from pnms import SimpleQueryEncoder

encoder = SimpleQueryEncoder(embed_dim=64, vocab_size=10000)
```

特点：

- 输入：字符串 → ASCII/Unicode 转 id → embedding → 平均池化。  
- 输出：固定维度向量，维度由 `embed_dim` 决定。

### 5.2 SentenceEncoder

基于 `sentence-transformers` 的语义编码器，适合生产环境：

```python
from pnms import SentenceEncoder
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceEncoder(device=device, model_name="paraphrase-multilingual-MiniLM-L12-v2")
```

特点：

- 输出维度由底层模型决定（如 384），`PNMS` 会自动同步到 `config.embed_dim`。  
- 推荐在高质量语义检索场景使用。

---

## 6. 异常与错误处理

所有 PNMS 相关异常均继承自 `PNMSError`：

- `ConfigError`：配置不合法。  
- `EncoderError`：编码器加载或前向失败。  
- `PersistenceError`：概念 / 图 / 状态的保存与加载失败。  
- `ConceptError`：概念形成或加载过程中的逻辑错误（如维度不一致）。  
- `GraphError`：图结构构建或查询错误（预留扩展）。  
- `StateError`：个人神经状态维度或序列化异常。  
- `LLMError`：LLM 回调失败（如网络错误、上游异常）。

`PNMSError` 支持可选字段 `code`，推荐调用方优先按错误码处理。与 **merge** 相关的错误码有：

- `ErrorCodes.MERGE_VERSION_INCOMPATIBLE`：源 checkpoint 记忆格式版本高于当前运行版本，拒绝合并。  
- `ErrorCodes.MERGE_INVALID_ARGUMENT`：参数格式非法、meta/graph 版本不一致、缺少可探测的版本、`memory_slots.json` 无法解析或槽与 ``embed_dim`` 不一致等。  
- `ErrorCodes.MERGE_CHECKPOINT_NOT_FOUND`：源 checkpoint 目录不存在。  
- `ErrorCodes.MERGE_NOT_IMPLEMENTED`：保留码；当前实现已完成合并逻辑，一般不应再出现。

典型捕获方式：

```python
from pnms import PNMSError, PersistenceError, LLMError

try:
    result = client.handle(user_id, query, llm=my_llm)
except LLMError:
    # LLM 调用失败，可降级为纯模板回复或返回错误信息
    ...
except PersistenceError:
    # 持久化失败，可降级为“无概念/无图模式”运行
    ...
except PNMSError:
    # 其他 PNMS 内部错误
    ...
```

---

## 7. 集成建议（面向 `openclaw-mmem` / `mindmemory`）

1. **初始化阶段**：  
   - 从配置文件或环境变量构造一个全局 `PNMSConfig`。  
   - 用此配置初始化单例 `PNMSClient`。

2. **每个用户的一轮对话**：  
   - 在记忆插件层中调用 `PNMSClient.handle`，将上层 LLM 函数传入。  
   - 使用 `HandleQueryResult.response` 作为用户可见回复，`context` / `num_slots_used` / `phase` 作为调试指标上报。

3. **服务关闭或定期**：  
   - 定期调用 `PNMS.save_concept_modules()`（或在服务关闭前）做一次 checkpoint，确保概念与图的状态持久化。

只要满足以上约定，`pnms` 可以作为一个相对独立的“记忆引擎模块”对外提供稳定接口，其内部实现（概念训练方式、图存储细节等）可以在不破坏集成代码的前提下继续演进。

