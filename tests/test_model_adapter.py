"""model_adapter.py 测试模块"""

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from src.adapters.model_adapter import LLMBackend, ModelConfig, NotSupportError


def make_config(
    chat_model: str = "chat-model-1",
    embed_model: str | None = "text-embedding-v1",
    supports_embeddings: bool = True,
) -> ModelConfig:
    """创建测试所用配置"""
    return ModelConfig(
        chat_model=chat_model,
        chat_api_key="sk-test",
        chat_base_url="http://test.local/v1",
        embed_model=embed_model,
        embed_api_key="sk-test",
        embed_base_url="http://test.local/v1",
        supports_chat=True,
        supports_embeddings=supports_embeddings,
    )


def test_config_is_frozen():
    """ModelConfig 是不可变的"""
    config = make_config()
    with pytest.raises(FrozenInstanceError):
        config.chat_model = "changed"


def test_config_independence():
    """两个 LLMBackend 的 config 互不影响"""
    backend_a = LLMBackend(make_config(chat_model="chat-model-1"))
    backend_b = LLMBackend(make_config(chat_model="chat-model-2"))

    assert backend_a.config.chat_model == "chat-model-1"
    assert backend_b.config.chat_model == "chat-model-2"
    assert backend_a.config is not backend_b.config


def test_supports_chat_false_fails_fast():
    """supports_chat=False 时，chat() / chat_message() 应 fail fast"""
    config = ModelConfig(
        chat_model=None,
        chat_api_key=None,
        chat_base_url=None,
        supports_chat=False,
        supports_embeddings=False,
    )
    backend = LLMBackend(config)

    with pytest.raises(NotSupportError, match="当前配置不支持chat调用"):
        asyncio.run(backend.chat_message([{"role": "user", "content": "hello"}]))

    with pytest.raises(NotSupportError, match="当前配置不支持chat调用"):
        asyncio.run(backend.chat([{"role": "user", "content": "hello"}]))


def test_supports_embeddings_false_fails_fast():
    """supports_embeddings=False 时，embed() 应 fail fast"""
    backend = LLMBackend(
        make_config(
            embed_model=None,
            supports_embeddings=False,
        )
    )

    with pytest.raises(NotSupportError, match="当前配置不支持embed调用"):
        asyncio.run(backend.embed(["hello"]))


def test_chat_client_created_when_supports_chat():
    """支持 chat 时会创建 chat_client"""
    config = make_config()
    backend = LLMBackend(config)

    assert backend.chat_client is not None


def test_embed_client_created_when_supports_embeddings():
    """支持 embeddings 时会创建 embed_client"""
    config = make_config()
    backend = LLMBackend(config)

    assert backend.embed_client is not None


def test_clients_are_none_when_not_supported():
    """不支持对应能力时，对应 client 应为 None"""
    config = ModelConfig(
        chat_model=None,
        chat_api_key=None,
        chat_base_url=None,
        embed_model=None,
        embed_api_key=None,
        embed_base_url=None,
        supports_chat=False,
        supports_embeddings=False,
    )
    backend = LLMBackend(config)

    assert backend.chat_client is None
    assert backend.embed_client is None


