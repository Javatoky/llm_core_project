import asyncio
import json
from collections.abc import Callable
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCall

from .errors import ToolArgumentsError, ToolExecutionError, ToolNotFoundError

ToolRegistry = dict[str, tuple[Callable[..., Any], list[str]]]

def wrap_ok_result(result: Any) -> dict:
    """包装成功结果"""
    return {"success": True, "data": result}

def wrap_fail_result(error: str) -> dict:
    """包装失败结果"""
    return {"success": False, "error": error}

def parse_tool_call(tool_call: ChatCompletionMessageToolCall) -> tuple[str, dict]:
    """工具请求解析"""    
    tool_name = tool_call.function.name
    
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        raise ToolArgumentsError(f"工具参数不是合法 JSON：{e}") from e
    
    if not isinstance(arguments, dict):
        raise ToolArgumentsError("工具参数必须是 JSON object")

    return tool_name, arguments

def route_tool(tool_name: str, tool_registry: dict[str, tuple[Callable, list[str]]]) -> tuple[Callable, list[str]]:
    """工具路由"""
    if (tool_info:=tool_registry.get(tool_name, None)) is None:
        raise ToolNotFoundError(f"未知工具名：{tool_name}")
    
    return tool_info

def validate_arguments(arguments: dict, required_args: list[str]) -> None:
    """参数检查"""
    missing_args = set()

    for arg in required_args:
        if arg not in arguments:
            missing_args.add(arg)

    extra_args = set(arguments) - set(required_args)

    if missing_args:
        raise ToolArgumentsError(f"参数缺失：{missing_args}")
    if extra_args:
        raise ToolArgumentsError(f"存在多余参数：{extra_args}")

async def invoke_tool(tool: Callable, arguments: dict) -> dict:
    """调用器"""
    try:
        return wrap_ok_result(await tool(**arguments))
    except ToolExecutionError as e:
        return wrap_fail_result(f"{e}")
    except Exception as e:
        return wrap_fail_result(f"工具执行失败：{type(e).__name__}: {e}")

def build_tool_message(tool_call_id: str, result: dict) -> dict:
    """标准化工具结果"""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(result, ensure_ascii = False)
    }

async def run_single_tool_call(tool_call: ChatCompletionMessageToolCall, tool_registry) -> tuple[str, dict, str | None]:
    """工具调用流"""
    try:
        tool_name, arguments = parse_tool_call(tool_call)

        tool, required_args = route_tool(tool_name, tool_registry)

        validate_arguments(arguments, required_args)
    except ToolNotFoundError as e:
        return tool_call.id, wrap_fail_result(f"工具名错误：{e}"), None
    except ToolArgumentsError as e:
        return tool_call.id, wrap_fail_result(f"参数错误：{e}"), None
    else:
        result = await invoke_tool(tool, arguments)

    return tool_call.id, result, tool_name
async def tool_call_pipeline(
    tool_calls: list[ChatCompletionMessageToolCall],
    tool_registry: dict[str, tuple[Callable, list[str]]],
) -> tuple[list[dict], list[str]]:
    """批量调用工具"""
        
    results = await asyncio.gather(
        *[
            run_single_tool_call(tool_call, tool_registry)
            for tool_call in tool_calls
        ]
    )

    tool_messages = [
        build_tool_message(tool_call_id, result)
        for tool_call_id, result, _ in results
    ]

    called_tools = [
        tool_name
        for _, _, tool_name in results
        if tool_name is not None
    ]

    return tool_messages, called_tools
