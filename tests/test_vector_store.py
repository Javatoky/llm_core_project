"""测试 vector_store.py 模块"""

from unittest.mock import MagicMock, patch

import pytest

from src.rag.vector_store import (get_or_create_collection, query_collection,
                                  upsert_chunks)


class TestUpsertChunks:
    """测试 upsert_chunks 函数"""

    def test_upsert_chunks_normal(self):
        """测试正常情况：chunks 和 embeddings 正确转换后传给 collection.upsert"""
        chunks = [
            {"id": "1", "text": "文本 1", "metadata": {"source": "doc1", "chunk_index": 0}},
            {"id": "2", "text": "文本 2", "metadata": {"source": "doc1", "chunk_index": 1}},
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        with patch("src.rag.vector_store.get_or_create_collection") as mock_get_collection:
            mock_collection = MagicMock()
            mock_get_collection.return_value = mock_collection

            upsert_chunks(chunks, embeddings, "test_collection")

            mock_collection.upsert.assert_called_once()
            call_args = mock_collection.upsert.call_args
            assert call_args.kwargs["ids"] == ["1", "2"]
            assert call_args.kwargs["documents"] == ["文本 1", "文本 2"]
            assert call_args.kwargs["metadatas"] == [
                {"source": "doc1", "chunk_index": 0},
                {"source": "doc1", "chunk_index": 1},
            ]
            assert call_args.kwargs["embeddings"] == embeddings

    def test_upsert_chunks_length_mismatch(self):
        """测试异常情况：chunks 和 embeddings 长度不一致时抛出 ValueError"""
        chunks = [
            {"id": "1", "text": "文本 1", "metadata": {}},
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        with pytest.raises(ValueError, match="chunks 和 embeddings 数量不一致"):
            upsert_chunks(chunks, embeddings, "test_collection")

    def test_upsert_chunks_empty(self):
        """测试空表插入"""
        chunks = []
        embeddings = []

        with patch("src.rag.vector_store.get_or_create_collection") as mock_get_collection:
            upsert_chunks(chunks, embeddings, "test_collection")
            mock_get_collection.assert_not_called()

class TestQueryCollection:
    """测试 query_collection 函数"""

    def test_query_collection_normal(self):
        """测试正常情况：调用 collection.query 并正确转换返回结果"""
        query_embedding = [0.5, 0.6]
        top_k = 2

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["文档 1", "文档 2"]],
            "metadatas": [[{"source": "doc1", "chunk_index": 0}, {"source": "doc1", "chunk_index": 1}]],
            "distances": [[0.1, 0.2]],
        }

        with patch("src.rag.vector_store._client.get_collection", return_value=mock_collection):
            results = query_collection(query_embedding, top_k, "test_collection")

            mock_collection.query.assert_called_once_with(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            assert len(results) == 2
            assert results[0] == {
                "reference_text": "文档 1",
                "source": "doc1",
                "chunk_index": 0,
                "distance": 0.1
            }
            assert results[1] == {
                "reference_text": "文档 2",
                "source": "doc1",
                "chunk_index": 1,
                "distance": 0.2
            }

    def test_query_collection_missing_metadata_fields(self):
        query_embedding = [0.5, 0.6]

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["文档 1"]],
            "metadatas": [[{}]],
            "distances": [[0.1]],
        }

        with patch("src.rag.vector_store._client.get_collection", return_value=mock_collection):
            results = query_collection(query_embedding, 1, "test_collection")

        assert results == [
            {
                "reference_text": "文档 1",
                "source": None,
                "chunk_index": None,
                "distance": 0.1,
            }
        ]

    def test_query_collection_not_found(self):
        """测试 Collection 不存在时抛出 ValueError"""
        query_embedding = [0.5, 0.6]

        def raise_error(name):
            raise ValueError("Collection not found")

        with patch("src.rag.vector_store._client.get_collection", side_effect=raise_error):
            with pytest.raises(ValueError, match="Collection not found"):
                query_collection(query_embedding, 2, "nonexistent_collection")
