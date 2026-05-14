import asyncio

import pytest

from src.adapters.model_adapter import LLMBackend, ModelConfig
from src.config import (QWEN_ALY_API_KEY, QWEN_ALY_BASE_URL, QWEN_API_KEY,
                        QWEN_BASE_URL, QWEN_EMBED_MODEL, QWEN_MODEL)


@pytest.mark.integration
def test_chat_integration():
    """集成测试模型chat接口"""
    model_config = ModelConfig(
        chat_model = QWEN_MODEL,
        chat_api_key = QWEN_API_KEY,
        chat_base_url = QWEN_BASE_URL,
        embed_model = QWEN_EMBED_MODEL,
        embed_api_key = QWEN_ALY_API_KEY,
        embed_base_url = QWEN_ALY_BASE_URL,
        supports_embeddings = True
    )

    backend = LLMBackend(model_config)
    
    messages = [
        {"role": "system", "content": ""}
    ]

    reply = asyncio.run(backend.chat(messages))
    
    assert isinstance(reply, str)
    assert reply.strip()

@pytest.mark.integration
def test_embed_integration():
    """集成测试模型embed接口"""
    model_config = ModelConfig(
        chat_model = QWEN_MODEL,
        chat_api_key = QWEN_API_KEY,
        chat_base_url = QWEN_BASE_URL,
        embed_model = QWEN_EMBED_MODEL,
        embed_api_key = QWEN_ALY_API_KEY,
        embed_base_url = QWEN_ALY_BASE_URL,
        supports_embeddings = True
    )

    backend = LLMBackend(model_config)

    test_content = "你好"

    vectors = asyncio.run(backend.embed([test_content]))

    assert isinstance(vectors, list)
    assert len(vectors) == 1
    assert isinstance(vectors[0], list)
    assert len(vectors[0]) > 0
    assert all(isinstance(x, float) for x in vectors[0][:10])