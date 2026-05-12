import asyncio

from ..adapters.model_adapter import LLMBackend, ModelConfig
from ..config import (QWEN_ALY_API_KEY, QWEN_ALY_BASE_URL, QWEN_API_KEY,
                      QWEN_BASE_URL, QWEN_EMBED_MODEL, QWEN_MODEL)
from ..prompts.hybrid_system_prompt import HYBRID_SYSTEM_PROMPT
from .hybrid_agent import hybrid_agent_reply


async def main():
    backend = LLMBackend(ModelConfig(
        chat_model = QWEN_MODEL,
        chat_api_key = QWEN_API_KEY,
        chat_base_url = QWEN_BASE_URL,
        embed_model = QWEN_EMBED_MODEL,
        embed_api_key = QWEN_ALY_API_KEY,
        embed_base_url = QWEN_ALY_BASE_URL,
        supports_embeddings = True
    ))

    messages = [{"role": "system", "content": HYBRID_SYSTEM_PROMPT}]

    result = await hybrid_agent_reply(
        backend=backend,
        user_input="帮我算一下 12 * 34",
        messages=messages,
    )

    print(result.answer)
    print(result.called_tools)
    print(messages)

if __name__ == "__main__":
    asyncio.run(main())