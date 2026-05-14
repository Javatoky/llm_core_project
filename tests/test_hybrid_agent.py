"""hybrid_agent.py 单元测试"""

import asyncio
import pytest
from unittest.mock import MagicMock

from src.agents.schemas import AgentResult
from src.agents import hybrid_agent


class TestHybridAgentReply:
    """hybrid_agent_reply() 函数测试"""

    def test_no_collection_name_disables_rag(self, monkeypatch):
        """不传 collection_name 时，不启用 RAG 工具"""
        called = {}

        def fake_build_tool_schemas(include_rag: bool = False):
            called["include_rag"] = include_rag
            return []

        def fake_build_tool_registry(search_documents=None):
            called["search_documents"] = search_documents
            return {}

        async def fake_agent_loop(**kwargs):
            return AgentResult(answer="最终回答", called_tools=[])

        monkeypatch.setattr(hybrid_agent, "build_tool_schemas", fake_build_tool_schemas)
        monkeypatch.setattr(hybrid_agent, "build_tool_registry", fake_build_tool_registry)
        monkeypatch.setattr(hybrid_agent, "agent_loop", fake_agent_loop)

        backend = MagicMock()
        messages = []

        asyncio.run(hybrid_agent.hybrid_agent_reply(
            backend=backend,
            user_input="你好",
            messages=messages,
            collection_name=None
        ))

        assert called["include_rag"] is False
        assert called["search_documents"] is None

    def test_with_collection_name_enables_rag(self, monkeypatch):
        """传入 collection_name 时，启用 search_documents 工具"""
        called = {}
        make_tool_called_with = {}

        def fake_build_tool_schemas(include_rag: bool = False):
            called["include_rag"] = include_rag
            return []

        def fake_build_tool_registry(search_documents=None):
            called["search_documents"] = search_documents
            return {}

        def fake_make_search_documents_tool(backend, collection_name):
            make_tool_called_with["backend"] = backend
            make_tool_called_with["collection_name"] = collection_name
            return MagicMock()

        async def fake_agent_loop(**kwargs):
            return AgentResult(answer="最终回答", called_tools=["search_documents"])

        monkeypatch.setattr(hybrid_agent, "build_tool_schemas", fake_build_tool_schemas)
        monkeypatch.setattr(hybrid_agent, "build_tool_registry", fake_build_tool_registry)
        monkeypatch.setattr(hybrid_agent, "agent_loop", fake_agent_loop)
        monkeypatch.setattr(hybrid_agent, "make_search_documents_tool", fake_make_search_documents_tool)

        backend = MagicMock()
        messages = []

        asyncio.run(hybrid_agent.hybrid_agent_reply(
            backend=backend,
            user_input="查询文档",
            messages=messages,
            collection_name="my_collection"
        ))

        assert called["include_rag"] is True
        assert called["search_documents"] is not None

        # 断言 make_search_documents_tool 收到了正确的参数
        assert make_tool_called_with["backend"] is backend
        assert make_tool_called_with["collection_name"] == "my_collection"

    def test_messages_only_contains_user_input_and_assistant_answer(self, monkeypatch):
        """长期 messages 只追加 user 原始输入和 assistant 最终回答"""
        called = {}

        def fake_build_tool_schemas(include_rag: bool = False):
            called["include_rag"] = include_rag
            return []

        def fake_build_tool_registry(search_documents=None):
            called["search_documents"] = search_documents
            return {}

        async def fake_agent_loop(backend, messages, tool_schemas, tool_registry):
            called["temp_messages"] = messages
            return AgentResult(answer="这是最终回答", called_tools=[])

        monkeypatch.setattr(hybrid_agent, "build_tool_schemas", fake_build_tool_schemas)
        monkeypatch.setattr(hybrid_agent, "build_tool_registry", fake_build_tool_registry)
        monkeypatch.setattr(hybrid_agent, "agent_loop", fake_agent_loop)

        backend = MagicMock()
        messages = [{"role": "user", "content": "之前的对话"}]

        asyncio.run(hybrid_agent.hybrid_agent_reply(
            backend=backend,
            user_input="你好",
            messages=messages,
            collection_name=None
        ))

        # 验证 messages 只增加了两条：user 输入和 assistant 回答
        assert len(messages) == 3
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"] == "你好"
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "这是最终回答"

    def test_returns_agent_result(self, monkeypatch):
        """返回值是 AgentResult，answer 和 called_tools 正确透传"""
        called = {}

        def fake_build_tool_schemas(include_rag: bool = False):
            called["include_rag"] = include_rag
            return []

        def fake_build_tool_registry(search_documents=None):
            called["search_documents"] = search_documents
            return {}

        async def fake_agent_loop(**kwargs):
            return AgentResult(
                answer="计算结果是 42",
                called_tools=["calculate", "get_weather"]
            )

        monkeypatch.setattr(hybrid_agent, "build_tool_schemas", fake_build_tool_schemas)
        monkeypatch.setattr(hybrid_agent, "build_tool_registry", fake_build_tool_registry)
        monkeypatch.setattr(hybrid_agent, "agent_loop", fake_agent_loop)

        backend = MagicMock()
        messages = []

        result = asyncio.run(hybrid_agent.hybrid_agent_reply(
            backend=backend,
            user_input="计算 21 * 2",
            messages=messages,
            collection_name=None
        ))

        assert result.answer == "计算结果是 42"
        assert result.called_tools == ["calculate", "get_weather"]

    def test_temp_messages_not_leaked_to_long_term_messages(self, monkeypatch):
        """不应把 tool messages / tool_calls 写入长期 messages"""
        called = {}

        def fake_build_tool_schemas(include_rag: bool = False):
            called["include_rag"] = include_rag
            return []

        def fake_build_tool_registry(search_documents=None):
            called["search_documents"] = search_documents
            return {}

        async def fake_agent_loop(backend, messages, tool_schemas, tool_registry):
            # 模拟 temp_messages 中包含 tool_calls 和 tool messages
            messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": {"name": "calculate"}}]})
            messages.append({"role": "tool", "content": "42", "tool_call_id": "call_1"})
            return AgentResult(answer="结果是 42", called_tools=["calculate"])

        monkeypatch.setattr(hybrid_agent, "build_tool_schemas", fake_build_tool_schemas)
        monkeypatch.setattr(hybrid_agent, "build_tool_registry", fake_build_tool_registry)
        monkeypatch.setattr(hybrid_agent, "agent_loop", fake_agent_loop)

        backend = MagicMock()
        messages = [{"role": "user", "content": "之前的对话"}]

        asyncio.run(hybrid_agent.hybrid_agent_reply(
            backend=backend,
            user_input="21 * 2 等于多少",
            messages=messages,
            collection_name=None
        ))

        # 验证长期 messages 中不应包含 tool 角色的消息
        roles_in_messages = [msg.get("role") for msg in messages]
        assert "tool" not in roles_in_messages

        # 验证最后两条消息是 user 输入和 assistant 回答
        assert messages[-2]["role"] == "user"
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "结果是 42"
