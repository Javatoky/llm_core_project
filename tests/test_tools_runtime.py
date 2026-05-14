"""runtime.py 测试模块"""

import asyncio
import json

import pytest
from openai.types.chat import ChatCompletionMessageToolCall

from src.tools.errors import (ToolArgumentsError, ToolExecutionError,
                              ToolNotFoundError)
from src.tools.runtime import (build_tool_message, invoke_tool,
                               parse_tool_call, route_tool, tool_call_pipeline,
                               validate_arguments)


def make_tool_call(
    tool_call_id: str = "call_123",
    tool_name: str = "test_tool",
    arguments: str = "{}",
) -> ChatCompletionMessageToolCall:
    """构造测试用的 ToolCall 对象"""
    return ChatCompletionMessageToolCall(
        id=tool_call_id,
        function={"name": tool_name, "arguments": arguments},
        type="function",
    )


# ==================== parse_tool_call 测试 ====================


class TestParseToolCall:
    """parse_tool_call() 测试"""

    def test_parse_valid_json(self):
        """能解析合法 JSON arguments"""
        tool_call = make_tool_call(arguments='{"a": 1, "b": 2}')
        name, args = parse_tool_call(tool_call)
        assert name == "test_tool"
        assert args == {"a": 1, "b": 2}

    def test_parse_empty_json(self):
        """能解析空 JSON object"""
        tool_call = make_tool_call(arguments="{}")
        name, args = parse_tool_call(tool_call)
        assert name == "test_tool"
        assert args == {}

    def test_parse_invalid_json_raises(self):
        """遇到非法 JSON 抛 ToolArgumentsError"""
        tool_call = make_tool_call(arguments='{"a": 1,}')
        with pytest.raises(ToolArgumentsError, match="不是合法 JSON"):
            parse_tool_call(tool_call)

    def test_parse_array_json_raises(self):
        """遇到非 object JSON（如数组）抛 ToolArgumentsError"""
        tool_call = make_tool_call(arguments="[]")
        with pytest.raises(ToolArgumentsError, match="必须是 JSON object"):
            parse_tool_call(tool_call)

    def test_parse_string_json_raises(self):
        """遇到字符串 JSON 抛 ToolArgumentsError"""
        tool_call = make_tool_call(arguments='"hello"')
        with pytest.raises(ToolArgumentsError, match="必须是 JSON object"):
            parse_tool_call(tool_call)

    def test_parse_number_json_raises(self):
        """遇到数字 JSON 抛 ToolArgumentsError"""
        tool_call = make_tool_call(arguments="123")
        with pytest.raises(ToolArgumentsError, match="必须是 JSON object"):
            parse_tool_call(tool_call)


# ==================== route_tool 测试 ====================


class TestRouteTool:
    """route_tool() 测试"""

    def test_find_registered_tool(self):
        """能找到已注册工具"""

        async def search(x: str) -> str:
            return x

        registry = {"search": (search, ["query"])}
        tool_func, required_args = route_tool("search", registry)
        assert tool_func is not None
        assert required_args == ["query"]

    def test_tool_not_found_raises(self):
        """找不到工具时抛 ToolNotFoundError"""

        async def search(x: str) -> str:
            return x

        registry = {"search": (search, ["query"])}
        with pytest.raises(ToolNotFoundError, match="未知工具名"):
            route_tool("unknown_tool", registry)


# ==================== validate_arguments 测试 ====================


class TestValidateArguments:
    """validate_arguments() 测试"""

    def test_valid_arguments(self):
        """参数正确时不抛异常"""
        validate_arguments({"a": 1, "b": 2}, ["a", "b"])

    def test_missing_argument_raises(self):
        """能识别缺失参数"""
        with pytest.raises(ToolArgumentsError, match="参数缺失"):
            validate_arguments({"a": 1}, ["a", "b"])

    def test_zero_not_missing(self):
        """不应把 0 误判为缺失"""
        validate_arguments({"count": 0}, ["count"])

    def test_false_not_missing(self):
        """不应把 False 误判为缺失"""
        validate_arguments({"flag": False}, ["flag"])

    def test_empty_string_not_missing(self):
        """不应把 "" 误判为缺失"""
        validate_arguments({"name": ""}, ["name"])

    def test_extra_argument_raises(self):
        """存在多余参数时抛异常"""
        with pytest.raises(ToolArgumentsError, match="存在多余参数"):
            validate_arguments({"a": 1, "b": 2, "c": 3}, ["a", "b"])


# ==================== invoke_tool 测试 ====================


class TestInvokeTool:
    """invoke_tool() 测试"""

    def test_success_result(self):
        """成功时包装 {"success": True, "data": ...}"""

        async def dummy_tool(x: int) -> int:
            return x * 2

        result = asyncio.run(invoke_tool(dummy_tool, {"x": 5}))
        assert result == {"success": True, "data": 10}

    def test_tool_execution_error(self):
        """捕获 ToolExecutionError 并包装 {"success": False, "error": ...}"""

        async def failing_tool():
            raise ToolExecutionError("工具执行失败")

        result = asyncio.run(invoke_tool(failing_tool, {}))
        assert result == {"success": False, "error": "工具执行失败"}

    def test_generic_exception(self):
        """捕获普通异常并包装"""

        async def error_tool():
            raise ValueError("未知错误")

        result = asyncio.run(invoke_tool(error_tool, {}))
        assert result["success"] is False
        assert "ValueError" in result["error"]


# ==================== build_tool_message 测试 ====================


class TestBuildToolMessage:
    """build_tool_message() 测试"""

    def test_content_is_json_string(self):
        """content 必须是 JSON 字符串"""
        result = {"success": True, "data": 42}
        message = build_tool_message("call_123", result)
        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call_123"
        assert isinstance(message["content"], str)
        assert json.loads(message["content"]) == result

    def test_content_with_chinese_chars(self):
        """content 支持中文"""
        result = {"success": True, "data": "你好"}
        message = build_tool_message("call_456", result)
        assert "你好" in message["content"]


# ==================== tool_call_pipeline 测试 ====================


class TestToolCallPipeline:
    """tool_call_pipeline() 测试"""

    def test_returns_tool_messages_and_called_tools(self):
        """返回 tool_messages 和 called_tools"""

        async def add(a: int, b: int) -> int:
            return a + b

        registry = {"add": (add, ["a", "b"])}

        tool_call = make_tool_call(
            tool_call_id="call_001",
            tool_name="add",
            arguments='{"a": 1, "b": 2}',
        )

        tool_messages, called_tools = asyncio.run(
            tool_call_pipeline([tool_call], registry)
        )

        assert isinstance(tool_messages, list)
        assert len(tool_messages) == 1
        assert tool_messages[0]["role"] == "tool"
        assert tool_messages[0]["tool_call_id"] == "call_001"
        assert called_tools == ["add"]

    def test_multiple_tool_calls(self):
        """批量调用多个工具"""

        async def add(a: int, b: int) -> int:
            return a + b

        async def mul(x: int, y: int) -> int:
            return x * y

        registry = {
            "add": (add, ["a", "b"]),
            "mul": (mul, ["x", "y"]),
        }

        tool_calls = [
            make_tool_call("call_1", "add", '{"a": 1, "b": 2}'),
            make_tool_call("call_2", "mul", '{"x": 3, "y": 4}'),
        ]

        tool_messages, called_tools = asyncio.run(
            tool_call_pipeline(tool_calls, registry)
        )

        assert len(tool_messages) == 2
        assert set(called_tools) == {"add", "mul"}

    def test_tool_not_found_in_pipeline(self):
        """工具未找到时仍返回消息"""

        async def add(a: int, b: int) -> int:
            return a + b

        registry = {"add": (add, ["a", "b"])}

        tool_call = make_tool_call(
            tool_call_id="call_1",
            tool_name="unknown",
            arguments='{}',
        )

        tool_messages, called_tools = asyncio.run(
            tool_call_pipeline([tool_call], registry)
        )

        assert len(tool_messages) == 1
        assert called_tools == []
        result = json.loads(tool_messages[0]["content"])
        assert result["success"] is False

    def test_invalid_arguments_in_pipeline(self):
        """参数错误时仍返回消息"""

        async def add(a: int, b: int) -> int:
            return a + b

        registry = {"add": (add, ["a", "b"])}

        tool_call = make_tool_call(
            tool_call_id="call_1",
            tool_name="add",
            arguments='{"a": 1}',  # 缺失 b
        )

        tool_messages, called_tools = asyncio.run(
            tool_call_pipeline([tool_call], registry)
        )

        assert len(tool_messages) == 1
        assert called_tools == []
        result = json.loads(tool_messages[0]["content"])
        assert result["success"] is False

    def test_invalid_json_in_pipeline(self):
        """arguments 不是合法 JSON 时仍返回失败 tool message"""

        async def add(a: int, b: int) -> int:
            return a + b

        registry = {"add": (add, ["a", "b"])}

        tool_call = make_tool_call(
            tool_call_id="call_1",
            tool_name="add",
            arguments='{"a": 1,}',
        )

        tool_messages, called_tools = asyncio.run(
            tool_call_pipeline([tool_call], registry)
        )

        assert called_tools == []
        assert len(tool_messages) == 1

        result = json.loads(tool_messages[0]["content"])
        assert result["success"] is False
        assert "JSON" in result["error"]

    def test_called_tools_includes_tool_when_execution_fails(self):
        """工具执行失败时，called_tools 仍记录已通过解析和参数校验的工具"""

        async def failing_tool(a: int) -> str:
            raise ToolExecutionError("失败")

        registry = {"fail": (failing_tool, ["a"])}

        tool_call = make_tool_call(
            tool_call_id="call_1",
            tool_name="fail",
            arguments='{"a": 1}',
        )

        tool_messages, called_tools = asyncio.run(
            tool_call_pipeline([tool_call], registry)
        )

        assert called_tools == ["fail"]

        result = json.loads(tool_messages[0]["content"])
        assert result["success"] is False
        assert "失败" in result["error"]
