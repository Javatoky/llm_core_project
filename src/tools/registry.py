"""工具注册表"""

from collections.abc import Callable

from src.tools.math_tools import calculate
from src.tools.weather_tools import get_weather

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

SEARCH_DOCUMENTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "在本地知识库、私有文档、笔记或参考资料中检索与用户问题相关的内容。"
            "当用户询问文档、提示词、笔记、资料内容时应使用此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用于检索本地知识库的查询语句，应保留用户问题中的关键信息。",
                }
            },
            "required": ["query"],
        },
    },
}



def build_tool_schemas(include_rag: bool = False) -> list[dict]:
    """返回当前可用工具的 schema"""
    schemas = [GET_WEATHER_SCHEMA, CALCULATE_SCHEMA]

    if include_rag:
        schemas.append(SEARCH_DOCUMENTS_SCHEMA)

    return schemas


def build_tool_registry(search_documents: Callable | None = None) -> dict[str, tuple[Callable, list[str]]]:
    """返回当前可用工具注册表"""
    registry = {
        "get_weather": (get_weather, ["city"]),
        "calculate": (calculate, ["expression"]),
    }

    if search_documents is not None:
        registry["search_documents"] = (search_documents, ["query"])

    return registry
