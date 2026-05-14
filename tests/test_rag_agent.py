"""rag_agent.py 测试模块

测试 src/rag/rag_agent.py 中的核心函数。
使用 monkeypatch 避免真实 ChromaDB 和模型调用。
"""
import asyncio

import pytest

from src.adapters.model_adapter import LLMBackend, ModelConfig
from src.rag.rag_agent import (
    RAG_SYSTEM_PROMPT,
    answer_with_rag,
    build_vector_store,
    format_context,
    retrieve,
)

# ==================== format_context 测试 ====================


def test_format_context_with_results():
    """测试 format_context() 能把检索结果拼成包含 source / chunk_index / text 的上下文"""
    texts = [
        {
            "source": "doc1.md",
            "chunk_index": 1,
            "reference_text": "这是第一段文本内容",
        },
        {
            "source": "doc2.md",
            "chunk_index": 2,
            "reference_text": "这是第二段文本内容",
        },
    ]

    result = format_context(texts)

    assert "[source]: doc1.md" in result
    assert "[chunk_index]: 1" in result
    assert "[text]: 这是第一段文本内容" in result
    assert "[source]: doc2.md" in result
    assert "[chunk_index]: 2" in result
    assert "[text]: 这是第二段文本内容" in result


def test_format_context_empty():
    """测试 format_context([]) 对空结果的行为"""
    result = format_context([])
    assert result == ""


def test_format_context_includes_distance():
    """测试 format_context() 包含 distance 信息"""
    texts = [
        {
            "source": "doc1.md",
            "chunk_index": 1,
            "reference_text": "相关内容",
            "distance": 0.35,
        },
    ]

    result = format_context(texts)

    assert "distance" in result or "[text]:" in result


# ==================== retrieve 测试 ====================


@pytest.fixture
def backend():
    """创建 mock backend"""
    config = ModelConfig(
        chat_model="qwen-plus",
        chat_api_key="test-key",
        chat_base_url="http://test.local/v1",
        embed_model="text-embedding-v3",
        embed_api_key="test-key",
        embed_base_url="http://test.local/v1",
        supports_embeddings=True,
        supports_chat=True,
    )
    return LLMBackend(config)


def test_retrieve_calls_embed(monkeypatch, backend):
    """测试 retrieve() 会调用 backend.embed([query])"""
    query = "测试查询"
    collection_name = "test_collection"
    mock_embedding = [0.1, 0.2, 0.3]

    async def mock_embed(texts):
        return [mock_embedding] * len(texts)

    monkeypatch.setattr(backend, "embed", mock_embed)

    mock_query_result = [
        {
            "reference_text": "匹配内容",
            "source": "doc.md",
            "chunk_index": 1,
            "distance": 0.5,
        }
    ]

    def mock_query_collection(query_embedding, top_k, collection_name):
        assert query_embedding == mock_embedding
        return mock_query_result

    monkeypatch.setattr(
        "src.rag.rag_agent.query_collection", mock_query_collection
    )

    result = asyncio.run(retrieve(backend, query, collection_name))

    assert len(result) == 1
    assert result[0]["reference_text"] == "匹配内容"


def test_retrieve_calls_query_collection_with_correct_params(monkeypatch, backend):
    """测试 retrieve() 会调用 query_collection(query_embedding=..., top_k=..., collection_name=...)"""
    query = "测试查询"
    collection_name = "my_docs"
    top_k = 5
    mock_embedding = [0.1, 0.2, 0.3]

    async def mock_embed(texts):
        return [mock_embedding] * len(texts)

    monkeypatch.setattr(backend, "embed", mock_embed)

    captured_params = {}

    def mock_query_collection(query_embedding, top_k, collection_name):
        captured_params["query_embedding"] = query_embedding
        captured_params["top_k"] = top_k
        captured_params["collection_name"] = collection_name
        return []

    monkeypatch.setattr(
        "src.rag.rag_agent.query_collection", mock_query_collection
    )

    asyncio.run(retrieve(backend, query, collection_name, top_k=top_k))

    assert captured_params["query_embedding"] == mock_embedding
    assert captured_params["top_k"] == top_k
    assert captured_params["collection_name"] == collection_name


def test_retrieve_filters_by_distance_threshold(monkeypatch, backend):
    """测试 retrieve() 会根据 distance_threshold 过滤结果"""
    query = "测试查询"
    collection_name = "test_collection"
    mock_embedding = [0.1, 0.2, 0.3]
    distance_threshold = 0.6

    async def mock_embed(texts):
        return [mock_embedding] * len(texts)

    monkeypatch.setattr(backend, "embed", mock_embed)

    mock_query_result = [
        {"reference_text": "内容 1", "source": "doc1.md", "chunk_index": 1, "distance": 0.3},
        {"reference_text": "内容 2", "source": "doc2.md", "chunk_index": 2, "distance": 0.5},
        {"reference_text": "内容 3", "source": "doc3.md", "chunk_index": 3, "distance": 0.7},
        {"reference_text": "内容 4", "source": "doc4.md", "chunk_index": 4, "distance": 0.9},
    ]

    def mock_query_collection(query_embedding, top_k, collection_name):
        return mock_query_result

    monkeypatch.setattr(
        "src.rag.rag_agent.query_collection", mock_query_collection
    )

    result = asyncio.run(
        retrieve(backend, query, collection_name, distance_threshold=distance_threshold)
    )

    assert len(result) == 2
    assert all(r["distance"] <= distance_threshold for r in result)


# ==================== answer_with_rag 测试 ====================


def test_answer_with_rag_retrieves_then_calls_chat(monkeypatch, backend):
    """测试 answer_with_rag() 会先 retrieve，再构造带 context 的 messages，再调用 backend.chat()"""
    user_input = "如何提升元认知能力？"
    collection_name = "test_collection"
    messages = []

    mock_embedding = [0.1, 0.2, 0.3]
    mock_retrieve_result = [
        {
            "reference_text": "元认知能力提升方法...",
            "source": "metacognitive.md",
            "chunk_index": 1,
            "distance": 0.4,
        }
    ]
    expected_reply = "根据资料，提升元认知能力的方法包括..."

    async def mock_embed(texts):
        return [mock_embedding] * len(texts)

    monkeypatch.setattr(backend, "embed", mock_embed)

    def mock_query_collection(query_embedding, top_k, collection_name):
        return mock_retrieve_result

    monkeypatch.setattr(
        "src.rag.rag_agent.query_collection", mock_query_collection
    )

    async def mock_chat(msgs, **kwargs):
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert RAG_SYSTEM_PROMPT in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert "【参考文本信息】" in msgs[1]["content"]
        assert "【用户问题】" in msgs[1]["content"]
        assert user_input in msgs[1]["content"]
        return expected_reply

    monkeypatch.setattr(backend, "chat", mock_chat)

    reply = asyncio.run(
        answer_with_rag(backend, user_input, collection_name, messages)
    )

    assert reply == expected_reply
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == user_input
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == expected_reply


def test_answer_with_rag_no_results_returns_fallback(monkeypatch, backend):
    """测试无检索结果时，answer_with_rag() 应返回固定拒答文案"""
    user_input = "一个找不到答案的问题"
    collection_name = "test_collection"
    messages = []

    mock_embedding = [0.1, 0.2, 0.3]
    expected_fallback = "根据现有资料无法回答此问题。"

    async def mock_embed(texts):
        return [mock_embedding] * len(texts)

    monkeypatch.setattr(backend, "embed", mock_embed)

    def mock_query_collection(query_embedding, top_k, collection_name):
        return []

    monkeypatch.setattr(
        "src.rag.rag_agent.query_collection", mock_query_collection
    )

    reply = asyncio.run(
        answer_with_rag(backend, user_input, collection_name, messages)
    )

    assert reply == expected_fallback
    assert len(messages) == 2
    assert messages[0]["content"] == user_input
    assert messages[1]["content"] == expected_fallback


def test_answer_with_rag_with_all_retrievals_filtered_out(monkeypatch, backend):
    """测试当所有检索结果都因距离阈值被过滤时，返回拒答文案"""
    user_input = "测试问题"
    collection_name = "test_collection"
    messages = []

    mock_embedding = [0.1, 0.2, 0.3]
    expected_fallback = "根据现有资料无法回答此问题。"

    async def mock_embed(texts):
        return [mock_embedding] * len(texts)

    monkeypatch.setattr(backend, "embed", mock_embed)

    def mock_query_collection(query_embedding, top_k, collection_name):
        return [
            {
                "reference_text": "相关内容",
                "source": "doc.md",
                "chunk_index": 1,
                "distance": 0.95,
            }
        ]

    monkeypatch.setattr(
        "src.rag.rag_agent.query_collection", mock_query_collection
    )

    reply = asyncio.run(
        answer_with_rag(backend, user_input, collection_name, messages)
    )

    assert reply == expected_fallback


# ==================== build_vector_store 测试 ====================


def test_build_vector_store_processes_documents_and_upserts(monkeypatch, tmp_path):
    """测试 build_vector_store 处理文档并存储"""
    test_file = tmp_path / "test_doc.md"
    test_file.write_text("# Test Document\n\nThis is test content.")

    config = ModelConfig(
        chat_model="qwen-plus",
        chat_api_key="test-key",
        chat_base_url="http://test.local/v1",
        embed_model="text-embedding-v3",
        embed_api_key="test-key",
        embed_base_url="http://test.local/v1",
        supports_embeddings=True,
    )
    test_backend = LLMBackend(config)

    mock_embedding = [0.1, 0.2, 0.3]

    async def mock_embed(texts):
        return [mock_embedding] * len(texts)

    monkeypatch.setattr(test_backend, "embed", mock_embed)

    captured_chunks = []
    captured_embeddings = []
    captured_collection_name = []

    def mock_upsert_chunks(chunks, embeddings, collection_name):
        captured_chunks.extend(chunks)
        captured_embeddings.extend(embeddings)
        captured_collection_name.append(collection_name)

    monkeypatch.setattr(
        "src.rag.rag_agent.upsert_chunks", mock_upsert_chunks
    )

    asyncio.run(build_vector_store(test_backend, [str(test_file)], "test_collection"))

    assert len(captured_chunks) > 0
    assert len(captured_embeddings) == len(captured_chunks)
    assert captured_collection_name == ["test_collection"]


def test_build_vector_store_empty_documents(monkeypatch, tmp_path):
    """测试 build_vector_store 处理空文档列表的行为"""
    config = ModelConfig(
        chat_model="qwen-plus",
        chat_api_key="test-key",
        chat_base_url="http://test.local/v1",
        embed_model="text-embedding-v3",
        embed_api_key="test-key",
        embed_base_url="http://test.local/v1",
        supports_embeddings=True,
    )
    test_backend = LLMBackend(config)

    embed_called = False

    async def mock_embed(texts):
        nonlocal embed_called
        embed_called = True
        return [[0.1, 0.2, 0.3]] * len(texts)

    monkeypatch.setattr(test_backend, "embed", mock_embed)

    upsert_called = False

    def mock_upsert_chunks(chunks, embeddings, collection_name):
        nonlocal upsert_called
        upsert_called = True

    monkeypatch.setattr(
        "src.rag.rag_agent.upsert_chunks", mock_upsert_chunks
    )

    asyncio.run(build_vector_store(test_backend, [], "test_collection"))

    assert embed_called is False
    assert upsert_called is False
