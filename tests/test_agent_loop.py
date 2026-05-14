"""agent_loop.py 单元测试"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from src.agents.agent_loop import agent_loop
from src.tools.runtime import ToolRegistry


class FakeBackend:
    """模拟 LLM 后端，返回预设的消息序列"""

    def __init__(self, messages_to_return: list):
        self.messages_to_return = messages_to_return
        self.call_count = 0
        self.calls: list[dict] = []

    async def chat_message(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.call_count >= len(self.messages_to_return):
            return MagicMock(content="", tool_calls=None)
        message = self.messages_to_return[self.call_count]
        self.call_count += 1
        return message


def create_tool_call(tool_call_id: str, name: str, arguments: dict) -> MagicMock:
    """创建模拟的 tool_call 对象"""
    tool_call = MagicMock()
    tool_call.id = tool_call_id
    tool_call.function.name = name
    tool_call.function.arguments = json.dumps(arguments)
    return tool_call


def create_chat_message(content: str = "", tool_calls: list = None) -> MagicMock:
    """创建模拟的 ChatCompletionMessage 对象"""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    return message


# 测试工具函数
async def add(a: int, b: int) -> int:
    return a + b


async def multiply(x: int, y: int) -> int:
    return x * y


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return {
        "add": (add, ["a", "b"]),
        "multiply": (multiply, ["x", "y"]),
    }


class TestAgentLoop:
    """agent_loop 函数测试类"""

    def test_model_returns_plain_text(self, tool_registry):
        """测试 1: 模型直接返回普通文本时，agent_loop 直接结束"""
        backend = FakeBackend([
            create_chat_message(content="你好，我是助手")
        ])

        initial_messages = [{"role": "user", "content": "你好"}]
        result = asyncio.run(agent_loop(
            backend=backend,
            messages=initial_messages,
            tool_schemas=[],
            tool_registry=tool_registry,
        ))

        assert result.answer == "你好，我是助手"
        assert result.called_tools == []
        assert backend.call_count == 1
        assert len(initial_messages) == 2
        assert initial_messages[1]["role"] == "assistant"

    def test_model_returns_single_tool_call(self, tool_registry):
        """测试 2: 模型返回一个 tool_call 时，agent_loop 能执行工具并继续拿到最终回答"""
        tool_call = create_tool_call("call_1", "add", {"a": 3, "b": 5})

        backend = FakeBackend([
            create_chat_message(content="", tool_calls=[tool_call]),
            create_chat_message(content="计算结果是 8")
        ])

        initial_messages = [{"role": "user", "content": "3+5 等于多少"}]
        result = asyncio.run(agent_loop(
            backend=backend,
            messages=initial_messages,
            tool_schemas=[],
            tool_registry=tool_registry,
        ))

        assert result.answer == "计算结果是 8"
        assert result.called_tools == ["add"]
        assert backend.call_count == 2
        assert len(initial_messages) == 4

    def test_model_returns_multiple_tool_calls(self, tool_registry):
        """测试 3: 模型返回多个 tool_calls 时，agent_loop 能批量处理"""
        tool_call_1 = create_tool_call("call_1", "add", {"a": 2, "b": 3})
        tool_call_2 = create_tool_call("call_2", "multiply", {"x": 4, "y": 5})

        backend = FakeBackend([
            create_chat_message(content="", tool_calls=[tool_call_1, tool_call_2]),
            create_chat_message(content="两个计算都完成了")
        ])

        initial_messages = [{"role": "user", "content": "计算 2+3 和 4*5"}]
        result = asyncio.run(agent_loop(
            backend=backend,
            messages=initial_messages,
            tool_schemas=[],
            tool_registry=tool_registry,
        ))

        assert result.answer == "两个计算都完成了"
        assert len(result.called_tools) == 2
        assert "add" in result.called_tools
        assert "multiply" in result.called_tools
        assert backend.call_count == 2
        assert len(initial_messages) == 5

    def test_reaches_max_steps(self, tool_registry):
        """测试 4: 达到 max_steps 后仍未得到最终文本时，应返回明确错误"""
        tool_call = create_tool_call("call_1", "add", {"a": 1, "b": 1})

        backend = FakeBackend([
            create_chat_message(content="", tool_calls=[tool_call])
            for _ in range(10)
        ])

        initial_messages = [{"role": "user", "content": "持续调用工具"}]
        result = asyncio.run(agent_loop(
            backend=backend,
            messages=initial_messages,
            tool_schemas=[],
            tool_registry=tool_registry,
            max_steps=3,
        ))

        assert result.answer == "工具调用轮次超过上限，已终止。"
        assert backend.call_count == 3
        assert len(result.called_tools) == 3

    def test_messages_contains_final_assistant_answer(self, tool_registry):
        """测试 5: 最终 messages 中应追加最终 assistant answer"""
        backend = FakeBackend([
            create_chat_message(content="这是最终回答")
        ])

        initial_messages = [{"role": "user", "content": "你好"}]
        result = asyncio.run(agent_loop(
            backend=backend,
            messages=initial_messages,
            tool_schemas=[],
            tool_registry=tool_registry,
        ))

        assert result.answer == "这是最终回答"
        assert initial_messages[-1]["role"] == "assistant"
        assert initial_messages[-1]["content"] == "这是最终回答"

    def test_work_messages_includes_tool_messages(self, tool_registry):
        """测试 6: 工作 messages 会包含 tool messages"""
        tool_call = create_tool_call("call_1", "add", {"a": 10, "b": 20})

        backend = FakeBackend([
            create_chat_message(content="", tool_calls=[tool_call]),
            create_chat_message(content="结果是 30")
        ])

        initial_messages = [{"role": "user", "content": "10+20"}]
        result = asyncio.run(agent_loop(
            backend=backend,
            messages=initial_messages,
            tool_schemas=[],
            tool_registry=tool_registry,
        ))

        assert result.answer == "结果是 30"
        roles_in_messages = [msg["role"] for msg in initial_messages]
        assert "tool" in roles_in_messages
        assert "assistant" in roles_in_messages
        assert initial_messages[-1]["role"] == "assistant"

    def test_tool_schemas_passed_to_backend(self, tool_registry):
        """测试 7: tool_schemas 应该传入 backend 的 chat_message 调用"""
        tool_call = create_tool_call("call_1", "add", {"a": 10, "b": 20})

        backend = FakeBackend([
            create_chat_message(content="", tool_calls=[tool_call]),
            create_chat_message(content="结果是 30")
        ])

        initial_messages = [{"role": "user", "content": "10+20"}]
        tool_schemas = [{"name": "add", "description": "加法工具", "parameters": {"type": "object"}}]

        asyncio.run(agent_loop(
            backend=backend,
            messages=initial_messages,
            tool_schemas=tool_schemas,
            tool_registry=tool_registry,
        ))

        # 验证每次调用都传入了 tool_schemas
        assert backend.call_count == 2
        for call in backend.calls:
            assert "tools" in call["kwargs"]
            assert call["kwargs"]["tools"] == tool_schemas
