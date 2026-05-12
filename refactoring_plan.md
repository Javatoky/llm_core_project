# 阶段 E 项目重构计划v3
 
## 1. 当前已有文件列表
- model_adapter.py：...
- tool_agent.py：...
- rag_agent.py：...
- hybrid_agent.py：...
- model_comparison.py：...
- config.py：...
- debug_config.py:  ...

## 2. 目标项目结构
llm_core_project/
├── adapters/
│   └── model_adapter.py
├── agents/
│   ├── agent_loop.py
│   ├── hybrid_agent.py
│   └── schemas.py
├── tools/
│   ├── registry.py
│   ├── runtime.py
│   ├── errors.py
│   ├── math_tools.py
│   ├── weather_tools.py
│   └── document_tools.py
├── rag/
│   ├── rag_agent.py
│   ├── document_pipeline.py
│   └── vector_store.py
├── prompts/
│   └── hybrid_system_prompt.py
├── evals/
│   └── model_comparison.py
├── tests/
│   ├── test_model_adapter.py
│   ├── test_agent_loop.py
│   ├── test_tools_runtime.py
│   ├── test_math_tools.py
│   ├── test_weather_tools.py
│   ├── test_document_tools.py
│   ├── test_rag_agent.py
│   ├── test_hybrid_agent.py
│   ├── test_model_adapter_integration.py
│   ├── test_rag_agent_integration.py
│   └── test_hybrid_agent_integration.py
├── test_documents/
├── chromadb/
├── config.py
├── debug_config.py
└── main.py

## 3. 模块职责划分
### adapters/model_adapter.py
负责什么：提供模型调用的chat（直接返回content），chat_message（返回message层级），chat_stream（流式输出，但暂不采用） ，embed（文本向量化） 
不负责什么：模型底层配置的更改能力，以及任何agent能力 

### agents/agent_loop.py  
负责什么：循环调用模型（涉及message编排）、维护助手message、控制工具调用循环 
不负责什么：工具运行，接触模型底层配置, 直接接触模型chat，工具的具体功能 

### agents/hybrid_agent.py
负责什么：提供支持RAG检索和工具调用的对话接口
不负责什么：具体的RAG流和工具调用循环 
### agents/schemas.py
负责什么：提供agent返回的数据结构

### tools/registry.py
负责什么：提供工具注册信息，作为外部调用接口
不负责什么：工具的具体实现
### tools/errors.py
负责什么：各个工具运行错误类型，隐藏底层错误类型从而避免暴露底层实现逻辑

### tools/math_tools.py
负责什么：提供数学相关工具 

### tools/weather_tools.py
负责什么：提供天气相关工具 

### tools/document_tools.py
负责什么：提供文档相关工具 
### tools/runtime.py  
负责什么：工具调用流（涉及请求解析、工具路由、批量运行、结果处理）  

### rag/rag_agent.py
负责什么：文档存储流程编排，检索流程编排
不负责什么：文本读取，文本分块，元数据处理，存入数据库等具体任务步骤的实现
### rag/document_pipeline.py
负责什么：文档存储流程中各步骤的具体实现,负责 read_file、chunking、metadata
### rag/vector_store.py
负责什么：检索流程中各步骤的具体实现,负责 ChromaDB collection、upsert、query、distance filtering

### prompts/hybrid_system_prompt.py 
负责什么：提供提示词（有hybrid的系统提示词，后续如果有调用模型的工具，也可以提供专属的工具提示词）  

### evals/model_comparison.py
负责什么：评估模型回复效果，对比采样
### config.py
负责什么：提供模型所需配置api_key, base_url, model（当前阶段暂时固定配置） 
### debug_config.py
负责测试文档路径、测试 collection_name、debug 开关、测试问题集、是否打印 called_tools


## 4. 函数迁移表
| 原位置 | 函数/变量 | 新位置 | 迁移理由 |
|---|---|---|---|
| tool_agent.py | safe_eval | tools/math_tools.py | agent 层不关注具体工具实现，工具实现应归入 tools 层 | 
| tool_agent.py | SAFE_OPERATORS | tools/math_tools.py | 同上 | 
| tool_agent.py | calculate | tools/math_tools.py | 同上 | 
| tool_agent.py | make_search_documents | tools/document_tools.py | 工具实现归入 tools 层，保持职责单一 | 
| tool_agent.py | get_weather | tools/weather_tools.py | 同上 | 

| tool_agent.py | TOOLS | tools/registry.py | 工具配置与注册由工具层统一管理，agent 只需引入注册表 | 
| tool_agent.py | TOOL_REGISTRY | tools/registry.py | 集中管理便于维护和修改 |
| tool_agent.py | tool_call_loop | agents/agent_loop.py | agents层集中管理交互相关流程，tools层集中负责工具调用，应分离 |
| tool_agent.py | tool_call_pipeline | tools/runtime.py | tool_agent脱离agent层，变成只负责工具运行流程的runtime模块，放入tools层 |
| rag_agent.py | 存储流步骤的具体实现相关函数与变量 | rag/document_store_steps.py | rag_agent层只负责流程编排和控制，不负责各步骤的具体实现，同时一并分离出来做rag层，不涉及交互流程的编排 |
| rag_agent.py | 检索流步骤的具体实现相关函数与变量 | rag/retrieve_steps.py | 同上 |
| debug_config.py | AgentResult | agents/schemas.py | agent返回的数据结构集中放在agents层 |  


## 5. 测试计划
unit测试（使用fake单独测试）：

- test_model_adapter.py：测试ModelConfig和LLMBackend的独立性，测试chat、chat_message接口在模型配置不支持chat调用的时候能否正常抛出异常
- test_agent_loop.py ：测试tool_call_loop.py能否按预期的运行，返回预期结果（利用假模型测试） 
- test_rag_agent.py：使用伪造模型测试build_vector_store 和 retrieve两个集成接口 
- test_rag_steps.py：分层测试：文本读取、分块处理、元数据处理、存库等具体步骤实现能否正常运行，能否正常抛出异常
- test_tools_runtime.py:  分层测试请求解析、工具路由、工具执行，以及测试最终接口tool_call_pipeline，测试他们能否正常运行、能否正常的抛出异常
- test_tools.py：测试各个工具函数的运行是否正常，异常能否正确的抛出
- test_hybrid_agent.py：测试对话接口能否去除中间信息只保留正常信息

integration 测试（使用真实api的灰度测试）：
- test_model_adapter_integration.py
- test_rag_integration.py
- test_hybrid_agent_integration.py
测试接上真实api的情况下能否正常流转

## 6. 重构顺序 

第 1 步：新增tools层级，按设计分层放置工具，新增registry.py，新增error.py放置各个工具的运行异常，tools层级内部对外抛出相应异常, 新增runtime.py管理工具执行流程
第 2 步：建立agent_loop.py，内部融入tool_call_loop 
第 3 步：整理rag层，保持通过backend.embed()调用embed 
第 4 步：重构hybrid_agent.py, 只保留业务入口，调用agent_loop，返回AgentResult
第 5 步：迁移 prompts/、evals/、main.py
第 6 步：补单元测试和集成测试

# 🛠️ 下一步任务

现在开始第 1 步重构：只做 tools 层，不碰 Agent Loop，不碰 RAG。

请你先提交这 4 个文件的代码：

tools/math_tools.py
tools/weather_tools.py
tools/errors.py
tools/registry.py

要求：

math_tools.py 包含 SAFE_OPERATORS、safe_eval()、calculate()，必须彻底禁止 eval()。

weather_tools.py 只放 get_weather()，继续用模拟天气即可。

errors.py 放工具层异常，例如 ToolExecutionError、ToolNotFoundError、ToolArgumentsError。先不要设计太复杂。

registry.py 放工具 schema 和基础工具注册表。search_documents 可以先预留，不急着接，因为它依赖 RAG 和 backend。

这一步目标是把纯工具层先迁稳。