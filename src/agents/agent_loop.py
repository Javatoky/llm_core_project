from openai.types.chat import ChatCompletionMessage

from src.adapters.model_adapter import LLMBackend
from src.agents.schemas import AgentResult
from src.tools.runtime import ToolRegistry, tool_call_pipeline


def build_assistant_tool_message(message: ChatCompletionMessage) -> dict:
    """标准处理模型返回的 message"""
    result = {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            tool_call.model_dump(mode = "json", exclude_none = True)
            for tool_call in message.tool_calls or []
        ]
    }

    return result

async def agent_loop(
    backend: LLMBackend,
    messages: list[dict],
    tool_schemas: list[dict],
    tool_registry: ToolRegistry,
    max_steps: int = 5,
) -> AgentResult:
    """工具调用循环"""
    called_tools: list[str] = []

    for _ in range(max_steps):
        
        message = await backend.chat_message(messages, tools = tool_schemas)

        if tool_calls:=message.tool_calls:
            messages.append(build_assistant_tool_message(message))

            tool_messages, partial_called_tools = await tool_call_pipeline(tool_calls, tool_registry)

            messages.extend(tool_messages)
            called_tools.extend(partial_called_tools)
        else:
            answer = message.content or ""
            messages.append({"role": "assistant", "content": answer})

            return AgentResult(answer, called_tools)
        
    return AgentResult(
        answer="工具调用轮次超过上限，已终止。",
        called_tools=called_tools,
    )
