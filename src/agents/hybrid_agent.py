import copy

from src.adapters.model_adapter import LLMBackend
from src.agents.agent_loop import agent_loop
from src.agents.schemas import AgentResult
from src.tools.document_tools import make_search_documents_tool
from src.tools.registry import build_tool_registry, build_tool_schemas


async def hybrid_agent_reply(
    backend: LLMBackend,
    user_input: str,
    messages: list[dict],
    collection_name: str | None = None
) -> AgentResult:
    """多工具智能助手入口, 只保存用户原始输入和最终回答；
    tool_calls / tool messages 只存在于 temp_messages 中。
    """
    if collection_name is not None:
        search_documents = make_search_documents_tool(backend, collection_name)
        tool_schemas = build_tool_schemas(include_rag = True)
        tool_registry = build_tool_registry(search_documents = search_documents)
    else:
        tool_schemas = build_tool_schemas()
        tool_registry = build_tool_registry()

    temp_messages = copy.deepcopy(messages)
    temp_messages.append({"role": "user", "content": user_input})
    
    result = await agent_loop(
        backend = backend,
        messages = temp_messages,
        tool_schemas = tool_schemas,
        tool_registry = tool_registry
    )

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": result.answer})
    
    return result