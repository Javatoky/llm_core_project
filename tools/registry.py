"""工具注册表"""

from collections.abc import Callable

from .math_tools import calculate
from .weather_tools import get_weather


GET_WEATHER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的当天天气。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如'北京'、'上海'。",
                }
            },
            "required": ["city"],
        },
    },
}

CALCULATE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "计算安全的数学表达式，只支持数字、括号和 + - * / 运算符。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如'1234 * 5678'、'(67 - 34) / 3'。",
                }
            },
            "required": ["expression"],
        },
    },
}


def build_tool_schemas() -> list[dict]:
    """返回当前可用工具的 schema"""
    return [GET_WEATHER_SCHEMA, CALCULATE_SCHEMA]


def build_tool_registry() -> dict[str, tuple[Callable, list[str]]]:
    """返回当前可用工具注册表"""
    return {
        "get_weather": (get_weather, ["city"]),
        "calculate": (calculate, ["expression"]),
    }
