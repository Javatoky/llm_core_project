import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.config import (QWEN_ALY_API_KEY, QWEN_ALY_BASE_URL, QWEN_API_KEY,
                        QWEN_BASE_URL, QWEN_EMBED_MODEL, QWEN_MODEL)

from src.adapters.model_adapter import LLMBackend, ModelConfig
from src.rag.rag_agent import answer_with_rag, build_vector_store

filepaths = [
    "test_documents/gemini_prompt.md",
    "test_documents/my_view_on_friendship.md",
    "test_documents/ai_and_coding.md",
    "test_documents/metacognitive_role.md",
    "test_documents/personal_prioritization_under_multiple_tasks.md"
]

# Resolve paths relative to project root
project_root = Path(__file__).resolve().parent.parent.parent
filepaths = [str(project_root / fp) for fp in filepaths]

test_cases = [
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

    await build_vector_store(
        backend=backend,
        filepaths=filepaths,
        collection_name="test_collection",
    )

    for i, case in enumerate(test_cases, 1):
        print(f"=== Test{i} ===")
        messages = []

        answer = await answer_with_rag(
            backend=backend,
            user_input=case,
            collection_name="test_collection",
            messages=messages,
        )

        print(answer)
        print(messages)

if __name__ == "__main__":
    asyncio.run(main())