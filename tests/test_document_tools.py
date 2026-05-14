"""document_tools.py 测试模块"""

import asyncio
import pytest

from src.tools.document_tools import make_search_documents_tool
from src.tools.errors import ToolExecutionError


# ==================== make_search_documents_tool 测试 ====================


class TestMakeSearchDocumentsTool:
    """make_search_documents_tool() 返回的 search_documents 函数测试"""

    def test_query_empty_string_raises(self, monkeypatch):
        """query 为空字符串时抛 ToolExecutionError"""

        async def should_not_be_called(*args, **kwargs):
            raise AssertionError("retrieve should not be called for empty query")

        monkeypatch.setattr("src.tools.document_tools.retrieve", should_not_be_called)

        search_documents = make_search_documents_tool(None, "test_collection")

        with pytest.raises(ToolExecutionError, match="查询语句不能为空"):
            asyncio.run(search_documents(""))

    def test_query_whitespace_only_raises(self, monkeypatch):
        """query 只有空白字符时抛 ToolExecutionError"""

        async def should_not_be_called(*args, **kwargs):
            raise AssertionError("retrieve should not be called for whitespace-only query")

        monkeypatch.setattr("src.tools.document_tools.retrieve", should_not_be_called)

        search_documents = make_search_documents_tool(None, "test_collection")

        with pytest.raises(ToolExecutionError, match="查询语句不能为空"):
            asyncio.run(search_documents("   \n\t  "))

    def test_retrieve_returns_results(self, monkeypatch):
        """retrieve() 返回结果时，search_documents 返回原始结果"""

        async def dummy_retrieve(backend, query, collection_name):
            assert query == "测试查询"
            assert collection_name == "test_collection"
            return [{"content": "文档 1", "score": 0.9}]

        monkeypatch.setattr("src.tools.document_tools.retrieve", dummy_retrieve)

        search_documents = make_search_documents_tool(None, "test_collection")
        result = asyncio.run(search_documents("测试查询"))

        assert result == [{"content": "文档 1", "score": 0.9}]

    def test_retrieve_returns_empty_list_raises(self, monkeypatch):
        """retrieve() 返回空列表时，抛 ToolExecutionError("知识库中没有找到相关资料")"""

        async def dummy_retrieve(backend, query, collection_name):
            return []

        monkeypatch.setattr("src.tools.document_tools.retrieve", dummy_retrieve)

        search_documents = make_search_documents_tool(None, "test_collection")

        with pytest.raises(ToolExecutionError, match="知识库中没有找到相关资料"):
            asyncio.run(search_documents("测试查询"))

    def test_retrieve_raises_wrapped_as_tool_execution_error(self, monkeypatch):
        """retrieve() 自身抛异常时，包装成 ToolExecutionError"""

        async def dummy_retrieve(backend, query, collection_name):
            raise ValueError("ChromaDB 连接失败")

        monkeypatch.setattr("src.tools.document_tools.retrieve", dummy_retrieve)

        search_documents = make_search_documents_tool(None, "test_collection")

        with pytest.raises(ToolExecutionError, match="查找出错") as exc_info:
            asyncio.run(search_documents("测试查询"))

        assert "ChromaDB 连接失败" in str(exc_info.value)

    def test_retrieve_called_with_correct_parameters(self, monkeypatch):
        """调用 retrieve() 时传入正确的 backend、query、collection_name"""

        captured_kwargs = {}

        async def dummy_retrieve(backend, query, collection_name):
            captured_kwargs["backend"] = backend
            captured_kwargs["query"] = query
            captured_kwargs["collection_name"] = collection_name
            return [{"content": "result"}]

        mock_backend = object()
        search_documents = make_search_documents_tool(mock_backend, "my_docs")

        monkeypatch.setattr("src.tools.document_tools.retrieve", dummy_retrieve)

        asyncio.run(search_documents("search term"))

        assert captured_kwargs["backend"] is mock_backend
        assert captured_kwargs["query"] == "search term"
        assert captured_kwargs["collection_name"] == "my_docs"
