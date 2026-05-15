import asyncio
import uuid

import pytest

from src.adapters.model_adapter import LLMBackend, ModelConfig
from src.config import (QWEN_ALY_API_KEY, QWEN_ALY_BASE_URL, QWEN_API_KEY,
                        QWEN_BASE_URL, QWEN_EMBED_MODEL, QWEN_MODEL)
from src.rag.document_pipeline import process_document
from src.rag.rag_agent import answer_with_rag, build_vector_store, retrieve
from src.rag.vector_store import query_collection, upsert_chunks


def generate_unique_collection_name(prefix: str = "test_collection") -> str:
    """生成唯一的集合名称，避免测试之间相互干扰"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def create_test_document(tmp_path, content: str, filename: str = "test_doc.txt") -> str:
    """创建临时测试文档"""
    filepath = tmp_path / filename
    filepath.write_text(content, encoding="utf-8")
    return str(filepath)


@pytest.mark.integration
class TestRAGIntegration:
    """RAG 集成测试：验证完整链路"""

    @pytest.fixture
    def llm_backend(self) -> LLMBackend:
        """创建 LLM 后端实例"""
        config = ModelConfig(
            chat_model = QWEN_MODEL,
            chat_api_key = QWEN_API_KEY,
            chat_base_url = QWEN_BASE_URL,
            embed_model = QWEN_EMBED_MODEL,
            embed_api_key = QWEN_ALY_API_KEY,
            embed_base_url = QWEN_ALY_BASE_URL,
            supports_embeddings = True
        )
        return LLMBackend(config)

    @pytest.fixture
    def collection_name(self) -> str:
        """生成唯一的集合名称"""
        return generate_unique_collection_name()

    @pytest.fixture
    def cleanup_collection(self, collection_name: str):
        """测试后清理集合"""
        yield
        try:
            import chromadb
            from src.config import CHROMA_PATH
            client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            client.delete_collection(name=collection_name)
        except Exception:
            pass  # 集合可能不存在，忽略错误

    def test_build_and_retrieve_vector_store(self, llm_backend, collection_name, cleanup_collection, tmp_path):
        # 显式使用 fixture 以避免警告
        _ = cleanup_collection
        """测试 1：构建临时知识库，验证 retrieve() 能检索到相关文档"""
        # 准备测试文档：包含特定内容的文本
        doc_content = """
        机器学习是人工智能的一个分支。

        深度学习使用神经网络来模拟人类大脑的工作方式。

        自然语言处理技术可以让计算机理解人类语言。
        """
        doc_path = create_test_document(tmp_path, doc_content)

        # 构建向量库
        asyncio.run(build_vector_store(llm_backend, [doc_path], collection_name))

        # 执行检索查询
        results = asyncio.run(retrieve(llm_backend, "什么是深度学习？", collection_name))

        # 验证检索结果
        assert len(results) > 0, "应该检索到相关文档"
        assert "reference_text" in results[0], "结果应包含参考文本"
        assert "source" in results[0], "结果应包含来源信息"
        assert "chunk_index" in results[0], "结果应包含分块索引"
        assert "distance" in results[0], "结果应包含距离分数"

        # 验证检索内容相关性
        retrieved_text = results[0]["reference_text"]
        assert "深度" in retrieved_text or "学习" in retrieved_text or "神经" in retrieved_text, \
            "检索结果应与查询相关"

    def test_answer_with_rag(self, llm_backend, collection_name, cleanup_collection, tmp_path):
        # 显式使用 fixture 以避免警告
        _ = cleanup_collection
        """测试 2：验证 answer_with_rag() 能基于检索结果返回非空回答，并写入 messages"""
        # 准备测试文档
        doc_content = """
        Python 是一种高级编程语言，由 Guido van Rossum 于 1989 年发明。

        Python 的设计哲学强调代码的可读性和简洁性。

        Python 广泛应用于 Web 开发、数据分析、人工智能等领域。
        """
        doc_path = create_test_document(tmp_path, doc_content)

        # 构建向量库
        asyncio.run(build_vector_store(llm_backend, [doc_path], collection_name))

        # 准备消息列表
        messages: list[dict] = []

        # 执行 RAG 问答
        reply = asyncio.run(
            answer_with_rag(
                backend=llm_backend,
                user_input="Python 是谁发明的？",
                collection_name=collection_name,
                messages=messages,
            )
        )

        # 验证返回结果
        assert reply is not None, "回答不应为空"
        assert len(reply) > 0, "回答应包含内容"

        # 验证 messages 被正确写入
        assert len(messages) == 2, "应该添加 user 和 assistant 两条消息"
        assert messages[0]["role"] == "user", "第一条消息应该是用户角色"
        assert messages[0]["content"] == "Python 是谁发明的？", "用户消息内容应该匹配"
        assert messages[1]["role"] == "assistant", "第二条消息应该是助手角色"
        assert messages[1]["content"] == reply, "助手消息内容应该等于返回的回答"

    def test_answer_with_no_results(self, llm_backend, collection_name, cleanup_collection, tmp_path):
        # 显式使用 fixture 以避免警告
        _ = cleanup_collection
        _ = tmp_path
        """测试 3：验证当检索不到结果时，answer_with_rag() 返回默认提示"""
        from src.rag.vector_store import get_or_create_collection

        # 创建一个空集合（不插入任何文档）
        get_or_create_collection(collection_name)

        messages: list[dict] = []

        # 执行 RAG 问答（查询不存在的内容）
        reply = asyncio.run(
            answer_with_rag(
                backend=llm_backend,
                user_input="量子纠缠如何影响区块链？",
                collection_name=collection_name,
                messages=messages,
            )
        )

        # 验证返回默认提示
        assert "无法回答" in reply or "无法回答此问题" in reply, \
            "应该返回无法回答的提示"

        # 验证 messages 被正确写入
        assert len(messages) == 2, "即使没有检索结果，也应该写入 user 和 assistant 消息"
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_process_document_integration(self, tmp_path):
        """测试 4：验证 process_document 能正确处理文档"""
        # 创建测试文档
        doc_content = "这是第一段。\n\n这是第二段。\n\n这是第三段。"
        doc_path = create_test_document(tmp_path, doc_content)

        # 处理文档
        chunks = process_document(doc_path, chunk_size=50, overlap=10)

        # 验证处理结果
        assert len(chunks) > 0, "应该生成至少一个分块"

        for chunk in chunks:
            assert "id" in chunk, "分块应包含 id"
            assert "text" in chunk, "分块应包含文本"
            assert "metadata" in chunk, "分块应包含元数据"
            assert "source" in chunk["metadata"], "元数据应包含来源"
            assert "chunk_index" in chunk["metadata"], "元数据应包含分块索引"

    def test_upsert_and_query_collection(self, collection_name, cleanup_collection):
        # 显式使用 fixture 以避免警告
        _ = cleanup_collection
        """测试 5：验证 upsert_chunks 和 query_collection 的基本功能"""
        # 准备测试数据
        chunks = [
            {
                "id": "test_1",
                "text": "人工智能是计算机科学的一个分支",
                "metadata": {"source": "test.txt", "chunk_index": 1}
            },
            {
                "id": "test_2",
                "text": "机器学习让计算机能够从数据中学习",
                "metadata": {"source": "test.txt", "chunk_index": 2}
            }
        ]

        # 模拟 embeddings（使用简单的向量）
        embeddings = [
            [0.1] * 1536,  # 假设维度为 1536
            [0.2] * 1536
        ]

        # 插入数据
        upsert_chunks(chunks, embeddings, collection_name)

        # 查询数据
        results = query_collection(
            query_embedding=[0.15] * 1536,
            top_k=2,
            collection_name=collection_name
        )

        # 验证查询结果
        assert len(results) > 0, "应该查询到结果"
        assert results[0]["reference_text"] in ["人工智能是计算机科学的一个分支", "机器学习让计算机能够从数据中学习"]
