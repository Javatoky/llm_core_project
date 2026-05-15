# test_hybrid_agent_integration.py 测试方针

## 测试内容
集成测试：src/agents/hybrid_agent.py

测试覆盖内容：
真实 hybrid_agent_reply()。

建议写 4 个 integration cases：

1. 普通问候：不应调用工具，called_tools == []
2. 数学计算：应调用 calculate，called_tools 包含 "calculate"
3. RAG 问题：传入 collection_name 后，应调用 search_documents，called_tools 包含 "search_documents"
4. 长期 messages 检查：多轮调用后，messages 中不应出现 role == "tool"，也不应出现 tool_calls 字段

其中 RAG case 需要先构建一个临时 collection

建议加 pytest.mark.integration，区分 unit 和 integration：

@pytest.mark.integration
def test_chat_integration():
    ...


## 代码编写细则
1. 使用 pytest assert 测试，在 test_hybrid_agent_integration.py 中完善测试内容
2. 我并未安装异步 pytest，所以对于异步函数的测试你可以使用 asyncio.run() 运行，或尝试其他能在普通测试函数运行的方法
3. 我已在文件中导入相应配置，你使用注入的配置即可
4. 这样只跑集成测试: pytest -m integration

## 问题补充
无
