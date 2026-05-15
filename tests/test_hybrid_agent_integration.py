import asyncio
import uuid

import pytest

from src.adapters.model_adapter import LLMBackend, ModelConfig
from src.agents.hybrid_agent import hybrid_agent_reply
from src.config import (QWEN_ALY_API_KEY, QWEN_ALY_BASE_URL, QWEN_API_KEY,
                        QWEN_BASE_URL, QWEN_EMBED_MODEL, QWEN_MODEL)
from src.rag.vector_store import upsert_chunks


def generate_unique_collection_name(prefix: str = "test_collection") -> str:
    """生成唯一的集合名称，避免测试之间相互干扰"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def llm_backend() -> LLMBackend:
    """创建 LLM 后端实例"""
    config = ModelConfig(
        chat_model=QWEN_MODEL,
        chat_api_key=QWEN_API_KEY,
        chat_base_url=QWEN_BASE_URL,
        embed_model=QWEN_EMBED_MODEL,
        embed_api_key=QWEN_ALY_API_KEY,
        embed_base_url=QWEN_ALY_BASE_URL,
        supports_embeddings=True
    )
    return LLMBackend(config)


@pytest.fixture
def collection_name() -> str:
    """生成唯一的集合名称"""
    return generate_unique_collection_name()


@pytest.fixture
def cleanup_collection(collection_name: str):
    """测试后清理集合"""
    yield
    try:
        import chromadb
        from src.config import CHROMA_PATH
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # 集合可能不存在，忽略错误


@pytest.mark.integration
class TestHybridAgentIntegration:
    """Hybrid Agent 集成测试：验证完整链路"""

    def test_ordinary_greeting(self, llm_backend):
        """测试 1：普通问候 - 不应调用工具，called_tools == []"""
        messages: list[dict] = []

        result = asyncio.run(
            hybrid_agent_reply(
                backend=llm_backend,
                user_input="你好，请介绍一下你自己",
                messages=messages,
            )
        )

        # 验证没有调用任何工具
        assert result.called_tools == [], f"普通问候不应调用工具，实际调用了：{result.called_tools}"

        # 验证返回了回答
        assert result.answer is not None, "回答不应为空"
        assert len(result.answer) > 0, "回答应包含内容"

        # 验证 messages 被正确写入
        assert len(messages) == 2, "应该添加 user 和 assistant 两条消息"
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "你好，请介绍一下你自己"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == result.answer

    def test_math_calculation(self, llm_backend):
        """测试 2：数学计算 - 应调用 calculate 工具"""
        messages: list[dict] = []

        result = asyncio.run(
            hybrid_agent_reply(
                backend=llm_backend,
                user_input="计算 23 加 47 等于多少",
                messages=messages,
            )
        )

        # 验证调用了 calculate 工具
        assert "calculate" in result.called_tools, \
            f"数学计算应调用 calculate 工具，实际调用了：{result.called_tools}"

        # 验证返回了回答
        assert result.answer is not None, "回答不应为空"
        assert len(result.answer) > 0, "回答应包含内容"

        # 验证 messages 被正确写入
        assert len(messages) == 2, "应该添加 user 和 assistant 两条消息"
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_rag_question(self, llm_backend, collection_name, cleanup_collection):
        """测试 3：RAG 问题 - 传入 collection_name 后，应调用 search_documents"""
        _ = cleanup_collection  # 显式使用 fixture 以避免警告

        # 准备测试数据：插入文档到向量库
        # 使用特定的虚构概念，确保模型无法用自己的知识回答，必须调用工具
        chunks = [
            {
                "id": "rag_test_1",
                "text": "XYZ-2077 是一种虚构的量子纠缠材料，由 Alpha 实验室在 2077 年首次合成。它具有在室温下实现超导的特性。",
                "metadata": {"source": "quantum_materials.txt", "chunk_index": 1}
            },
            {
                "id": "rag_test_2",
                "text": "XYZ-2077 材料的主要应用包括：量子计算机芯片、超高速磁悬浮列车和便携式核聚变反应堆。",
                "metadata": {"source": "quantum_materials.txt", "chunk_index": 2}
            }
        ]

        # 使用真实的 embeddings（通过 embedding 模型生成）
        texts = [chunk["text"] for chunk in chunks]
        embeddings = asyncio.run(llm_backend.embed(texts))

        # 插入数据到集合
        upsert_chunks(chunks, embeddings, collection_name)

        messages: list[dict] = []

        result = asyncio.run(
            hybrid_agent_reply(
                backend=llm_backend,
                user_input="XYZ-2077 材料有哪些主要应用？",
                messages=messages,
                collection_name=collection_name,
            )
        )

        # 验证调用了 search_documents 工具
        assert "search_documents" in result.called_tools, \
            f"RAG 问题应调用 search_documents 工具，实际调用了：{result.called_tools}"

        # 验证返回了回答
        assert result.answer is not None, "回答不应为空"
        assert len(result.answer) > 0, "回答应包含内容"

        # 验证 messages 被正确写入
        assert len(messages) == 2, "应该添加 user 和 assistant 两条消息"
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_long_term_messages(self, llm_backend):
        """测试 4：长期 messages 检查 - 多轮调用后，messages 中不应出现 role == "tool" 或 tool_calls 字段"""
        messages: list[dict] = []

        # 第一轮对话
        asyncio.run(
            hybrid_agent_reply(
                backend=llm_backend,
                user_input="计算 100 除以 4",
                messages=messages,
            )
        )

        # 第二轮对话
        asyncio.run(
            hybrid_agent_reply(
                backend=llm_backend,
                user_input="你好",
                messages=messages,
            )
        )

        # 第三轮对话
        asyncio.run(
            hybrid_agent_reply(
                backend=llm_backend,
                user_input="谢谢",
                messages=messages,
            )
        )

        # 验证 messages 中不应出现 role == "tool"
        for msg in messages:
            assert msg.get("role") != "tool", \
                f"messages 中不应出现 role == 'tool' 的消息：{msg}"

        # 验证 messages 中不应出现 tool_calls 字段
        for msg in messages:
            assert "tool_calls" not in msg, \
                f"messages 中不应出现 tool_calls 字段：{msg}"

        # 验证 messages 只包含 user 和 assistant 角色
        roles = [msg.get("role") for msg in messages]
        assert all(role in ["user", "assistant"] for role in roles), \
            f"messages 中只应包含 user 和 assistant 角色，实际包含：{set(roles)}"
