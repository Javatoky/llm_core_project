import copy

from ..adapters.model_adapter import LLMBackend
from ..tools.registry import build_tool_registry, build_tool_schemas
from .agent_loop import agent_loop
from .schemas import AgentResult


async def hybrid_agent_reply(
    backend: LLMBackend,
    user_input: str,
    messages: list[dict],
) -> AgentResult:
    """多工具智能助手入口, 只保存用户原始输入和最终回答；
    tool_calls / tool messages 只存在于 temp_messages 中。
    """

    temp_messages = copy.deepcopy(messages)
    temp_messages.append({"role": "user", "content": user_input})
    
    result = await agent_loop(
        backend = backend,
        messages = temp_messages,
        tool_schemas = build_tool_schemas(),
        tool_registry = build_tool_registry()
    )

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": result.answer})
    
    return result