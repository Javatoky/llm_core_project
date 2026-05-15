# Multi-Model RAG + Tool Use Assistant

## 项目简介

这是一个手写实现的 Multi-Model RAG + Tool Use Assistant 项目，目标是把 Prompt Engineering、Structured Output、Function Calling、Embedding 检索和 RAG 链路整合到一个可测试、可扩展的 LLM 应用架构中。

项目不依赖 LangChain、LlamaIndex、Dify 等编排框架，而是手写模型适配层、Tool Call Loop、工具运行时、RAG 文档处理管线和 Hybrid Agent，以便理解 LLM 应用开发的底层机制。

## 核心能力

### 1. 模型适配层

- 使用 `ModelConfig` 描述 chat / embedding 模型配置
- 使用 `LLMBackend` 统一封装 `chat()`、`chat_message()`、`chat_stream()`、`embed()`
- 支持 chat 模型和 embedding 模型分开配置
- 通过能力声明实现 fail fast，例如不支持 embedding 时直接抛出异常

### 2. Tool Use

- 支持 OpenAI-compatible Function Calling
- 实现通用 `agent_loop()`，支持多轮 tool call loop
- 支持 parallel tool calls 的批量执行
- 工具运行时负责解析 tool_call、路由工具、校验参数、包装工具结果

当前内置工具：

- `calculate`：基于 AST 白名单的安全数学计算工具
- `get_weather`：天气查询工具，目前为模拟数据
- `search_documents`：本地知识库检索工具

### 3. RAG 知识库

- 使用 ChromaDB 作为向量数据库
- 支持 Markdown / TXT 文档读取、分块、metadata 标注
- 使用 embedding 模型向量化文档 chunk
- 支持基于 query embedding 的 Top-K 相似度检索
- 支持将 RAG 作为 `search_documents` 工具接入 Hybrid Agent

### 4. 消息管理

- `agent_loop()` 使用工作 messages 处理 tool_calls / tool messages
- `hybrid_agent_reply()` 只把用户原始输入和最终回答写入长期 messages
- 避免 tool result 和 RAG context 污染长期对话历史

## 项目结构

```
llm_core_project/
├── src/                           # 主源码目录
│   ├── adapters/
│   │   └── model_adapter.py       # 模型调用适配层 (chat, chat_message, embed)
│   ├── agents/
│   │   ├── agent_loop.py          # 工具调用循环与模型交互编排
│   │   ├── hybrid_agent.py        # 支持 RAG 和工具调用的对话接口
│   │   └── schemas.py             # Agent 返回数据结构定义
│   ├── tools/
│   │   ├── registry.py            # 工具注册表与 schema 定义
│   │   ├── runtime.py             # 工具调用运行时 (请求解析、路由、批量执行)
│   │   ├── errors.py              # 工具层异常定义
│   │   ├── math_tools.py          # 数学计算工具 (safe_eval, calculate)
│   │   ├── weather_tools.py       # 天气查询工具
│   │   └── document_tools.py      # 文档检索工具
│   ├── rag/
│   │   ├── rag_agent.py           # RAG 流程编排 (存储流、检索流)
│   │   ├── document_pipeline.py   # 文档处理管道 (读取、分块、元数据)
│   │   └── vector_store.py        # 向量数据库操作 (ChromaDB)
│   ├── prompts/
│   │   └── hybrid_system_prompt.py # 系统提示词
│   ├── config.py                  # 模型配置 (api_key, base_url, model)
│   └── __init__.py
├── tests/                         # 测试目录
│   ├── test_model_adapter.py      # 模型适配层单元测试
│   ├── test_agent_loop.py         # Agent 循环单元测试
│   ├── test_tools_runtime.py      # 工具运行时单元测试
│   ├── test_math_tools.py         # 数学工具单元测试
│   ├── test_document_tools.py     # 文档工具单元测试
│   ├── test_rag_agent.py          # RAG Agent 单元测试
│   ├── test_hybrid_agent.py       # Hybrid Agent 单元测试
│   ├── test_registry.py           # 工具注册表测试
│   ├── test_vector_store.py       # 向量存储测试
│   ├── test_model_adapter_integration.py  # 模型适配层集成测试
│   ├── test_rag_agent_integration.py      # RAG Agent 集成测试
│   └── test_hybrid_agent_integration.py   # Hybrid Agent 集成测试
├── test_documents/                # 测试文档目录
├── chromadb/                      # ChromaDB 数据目录
├── .env                           # 环境变量配置
├── .env.example                   # 环境变量模板
└── requirements.txt               # 依赖清单
```

## 架构设计

### 分层职责

| 层级 | 模块 | 职责 | 不负责 |
|------|------|------|--------|
| **adapters** | `model_adapter.py` | 统一封装 chat / embedding API 调用，持有 `ModelConfig` 和 client | 读取环境变量、Agent 编排逻辑 |
| **agents** | `agent_loop.py` | 循环调用模型、维护消息编排、控制工具调用轮次 | 工具具体实现、直接接触模型底层 |
| **agents** | `hybrid_agent.py` | 提供对话入口，整合 RAG 和工具调用能力 | 具体的 RAG 流程、工具调用循环 |
| **agents** | `schemas.py` | 定义 Agent 返回数据结构 | 业务逻辑 |
| **tools** | `registry.py` | 工具 Schema 定义和注册表管理 | 工具具体实现 |
| **tools** | `runtime.py` | 工具调用运行时：请求解析、路由、批量执行、结果处理 | 工具业务逻辑 |
| **tools** | `errors.py` | 定义工具层异常类型 | 业务异常处理 |
| **tools** | `math_tools.py` | 数学计算工具实现 | 表达式解析以外的逻辑 |
| **tools** | `weather_tools.py` | 天气查询工具实现 | 真实天气 API 对接 |
| **tools** | `document_tools.py` | 文档检索工具封装 | 向量检索具体实现 |
| **rag** | `rag_agent.py` | RAG 流程编排：构建向量库、检索、基于上下文生成回答 | 文件读取、文本分块、ChromaDB 底层操作 |
| **rag** | `document_pipeline.py` | 文档处理：读取、分块、元数据标注 | 流程编排 |
| **rag** | `vector_store.py` | ChromaDB Collection 管理、chunk upsert、向量查询结果格式化 | 文档分块、RAG prompt 构造、距离阈值策略 |
| **prompts** | `hybrid_system_prompt.py` | 系统提示词模板 | 运行时逻辑 |

### 数据流

```
用户输入 → hybrid_agent_reply()
    ↓
    ├─→ (可选) 注册 search_documents 工具
    ↓
    └─→ agent_loop()
            ↓
            ├─→ backend.chat_message(messages, tools) → ChatCompletionMessage
            ↓
            ├─→ 有 tool_calls?
            │       ↓
            │       ├─→ tool_call_pipeline() → 执行工具 → tool_messages
            │       ↓
            │       └─→ 追加消息，继续下一轮
            ↓
            └─→ 无 tool_calls? → 返回 answer
```

### 设计原则

1. **模型调用与业务逻辑分离**：`LLMBackend` 只负责模型 API 调用，Agent 层不直接依赖具体 SDK client。
2. **工具定义与工具执行分离**：`registry.py` 管理 tool schema 和 registry，`runtime.py` 负责解析、路由、执行和结果包装。
3. **RAG 检索与 Agent Loop 解耦**：RAG 可以作为独立问答链路使用，也可以通过 `search_documents` 工具接入 Hybrid Agent。
4. **长期消息与工作消息分离**：tool_calls、tool messages 和 RAG context 只存在于临时工作 messages，不进入长期 messages。
5. **安全逻辑使用确定性代码实现**：数学工具使用 AST 白名单解析表达式，不依赖模型判断表达式是否安全。

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 API 密钥：

```bash
# 通义千问配置（chat 模型）
QWEN_API_KEY=your_api_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=your_chat_model_name

# 通义 embedding 模型配置
QWEN_ALY_API_KEY=your_api_key
QWEN_ALY_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_EMBED_MODEL=your_embed_model_name

# 其他模型配置（可选）
DS_API_KEY=your_api_key
DS_BASE_URL=your_deepseek_base_url
DS_MODEL=your_chat_model_name
```

## 运行方式

### 方式 1：直接调用 Python API

```python
import asyncio
from src.adapters.model_adapter import LLMBackend, ModelConfig
from src.agents.hybrid_agent import hybrid_agent_reply
from src.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL

config = ModelConfig(
    chat_model=QWEN_MODEL,
    chat_api_key=QWEN_API_KEY,
    chat_base_url=QWEN_BASE_URL,
    supports_chat=True
)

backend = LLMBackend(config)
messages = []

result = asyncio.run(
    hybrid_agent_reply(
        backend=backend,
        user_input="计算 23 加 47 等于多少",
        messages=messages
    )
)

print(result.answer)        # 助手回答
print(result.called_tools)  # 调用的工具列表
```

### 方式 2：使用 RAG 知识库

```python
import asyncio

from src.adapters.model_adapter import LLMBackend, ModelConfig
from src.agents.hybrid_agent import hybrid_agent_reply
from src.config import (
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL,
    QWEN_ALY_API_KEY,
    QWEN_ALY_BASE_URL,
    QWEN_EMBED_MODEL,
)
from src.rag.rag_agent import build_vector_store

config = ModelConfig(
    chat_model=QWEN_MODEL,
    chat_api_key=QWEN_API_KEY,
    chat_base_url=QWEN_BASE_URL,
    embed_model=QWEN_EMBED_MODEL,
    embed_api_key=QWEN_ALY_API_KEY,
    embed_base_url=QWEN_ALY_BASE_URL,
    supports_chat=True,
    supports_embeddings=True,
)

backend = LLMBackend(config)
messages = []

filepaths = ["docs/my_notes.md", "docs/project_info.md"]
collection_name = "my_knowledge_base"

asyncio.run(build_vector_store(backend, filepaths, collection_name))

result = asyncio.run(
    hybrid_agent_reply(
        backend=backend,
        user_input="请根据知识库回答：项目中的 Hybrid Agent 如何管理长期 messages？"
        messages=messages,
        collection_name=collection_name,
    )
)

print(result.answer)
print(result.called_tools)
```

## 测试方式

本项目测试分为 unit tests 和 integration tests。

### 运行单元测试

单元测试不调用真实模型 API，也不依赖真实外部服务：

```bash
pytest tests/ -m "not integration" -v
```
### 运行集成测试

集成测试会调用真实 LLM API、embedding API 和 ChromaDB：

```bash
pytest tests/ -m integration -v
```

### 运行全部测试

```bash
pytest tests/ -v
```

### pytest marker 配置

建议在 `pytest.ini` 中注册 integration marker：

```ini
[pytest]
markers =
    integration: tests that call real external APIs, model providers, or persistent vector stores
```

## 当前测试结果
当前测试共覆盖 138 个单元测试和 11 个集成测试：

| 类型 | 文件 | 结果 |
|------|------|------|
| Unit | `test_math_tools.py` | 42 passed |
| Unit | `test_tools_runtime.py` | 25 passed |
| Unit | `test_agent_loop.py` | 7 passed |
| Unit | `test_hybrid_agent.py` | 5 passed |
| Unit | `test_model_adapter.py` | 8 passed |
| Unit | `test_rag_agent.py` | 10 passed |
| Unit | `test_registry.py` | 6 passed |
| Unit | `test_document_tools.py` | 6 passed |
| Unit | `test_vector_store.py` | 6 passed |
| Unit | `test_document_pipeline.py` | 17 passed |
| Integration | `test_model_adapter_integration.py` | 2 passed |
| Integration | `test_rag_agent_integration.py` | 5 passed |
| Integration | `test_hybrid_agent_integration.py` | 4 passed |

## 当前限制

1. **天气工具仍是模拟实现**：`get_weather` 当前返回固定天气数据，尚未接入真实天气 API。
2. **Agent Loop 暂不支持流式最终回答**：`LLMBackend.chat_stream()` 已实现，但当前 `agent_loop()` 主路径仍使用非流式 `chat_message()`。
3. **模型切换方式较基础**：当前推荐通过创建新的 `LLMBackend(ModelConfig)` 实例切换模型，不支持在同一个 backend 实例上动态修改配置。
4. **RAG 检索策略较简单**：当前主要使用 Top-K + distance threshold，尚未接入 reranking。
5. **缺少用户交互入口**：当前主要通过 Python API 调用，尚未提供 CLI / Web UI。
6. **天气、搜索、文件等工具数量有限**：目前只内置 calculate、get_weather、search_documents 三类工具。

## 后续改进方向

1. **接入真实天气 API**：将 `get_weather` 从模拟实现替换为真实 API 调用。
2. **加入 reranking**：在向量检索后加入重排序，提高 RAG 检索质量。
3. **支持流式回答**：让 Hybrid Agent 支持最终回答的 streaming 输出。
4. **增加 CLI / Web UI**：提供更方便的用户交互入口。
5. **实现模型行为评估脚本**：对比不同模型在 Tool Use、RAG、Structured Output 上的行为差异。
6. **扩展工具系统**：增加文件读写、代码分析、网页检索等工具。
7. **增强长期记忆管理**：加入消息压缩、摘要记忆或按轮次裁剪策略。

## 项目价值定位
这个项目的重点不是堆叠工具数量，而是验证 LLM 应用中的几个关键工程边界：模型适配、工具调用协议、RAG 检索链路、消息隔离和分层测试。