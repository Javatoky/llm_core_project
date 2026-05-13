import asyncio
import os
import sys

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.adapters.model_adapter import LLMBackend, ModelConfig
from src.agents.hybrid_agent import hybrid_agent_reply
from src.config import (QWEN_ALY_API_KEY, QWEN_ALY_BASE_URL, QWEN_API_KEY,
                        QWEN_BASE_URL, QWEN_EMBED_MODEL, QWEN_MODEL)
from src.prompts.hybrid_system_prompt import HYBRID_SYSTEM_PROMPT

test_cases = [
    "Gemini 3提示词中给 Gemini 设定的 Goal 是什么？",
    # 预期：called_tools 包含 search_documents，回答标注 source/chunk_index

    "帮我算一下 789 * 321",
    # 预期：called_tools 包含 calculate

    "Trump 最喜欢的食物是什么？",
    # 预期：理想状态是 called_tools 包含 search_documents，然后回答“根据现有资料无法回答此问题。”
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
        print(f"\n=== Test{i} ===")
        messages = [{"role": "system", "content": HYBRID_SYSTEM_PROMPT}]

        result = await hybrid_agent_reply(
            backend=backend,
            user_input=case,
            messages=messages,
            collection_name="test_collection"
        )

        print(result.answer)
        print(result.called_tools)

if __name__ == "__main__":
    asyncio.run(main())