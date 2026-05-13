import asyncio
import sys
from pathlib import Path

from ..src.adapters.model_adapter import LLMBackend, ModelConfig
from ..src.agents.hybrid_agent import hybrid_agent_reply

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (QWEN_ALY_API_KEY, QWEN_ALY_BASE_URL, QWEN_API_KEY,
                    QWEN_BASE_URL, QWEN_EMBED_MODEL, QWEN_MODEL)

test_cases = [
    "你好，你能做什么？",
    "北京今天天气怎么样？",
    "帮我算一下 789 * 321",
    "Gemini 3提示词中给Gemini设定的Goal是什么？",
    "Trump 最喜欢的食物是什么？",
]

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

    for i, case in enumerate(test_cases, 1):
        print(f"\n---Test{i}---")
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        print(f"用户：{case}")
        reply = await hybrid_agent_reply(backend, case, messages, "my_documents")
        print(f"助手：{reply}")

if __name__ == "__main__":
    asyncio.run(main())