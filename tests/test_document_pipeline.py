"""测试 document_pipeline.py 模块"""

import os
import tempfile

import pytest

from src.rag.document_pipeline import (add_metadata, chunk_by_size, get_chunks,
                                       process_document, read_file)


class TestReadFile:
    """read_file() 测试"""

    def test_read_utf8_text_file(self):
        """能读取 UTF-8 文本文件"""
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as f:
            f.write("你好，世界！\n这是第二行。")
            temp_path = f.name

        try:
            content = read_file(temp_path)
            assert content == "你好，世界！\n这是第二行。"
        finally:
            os.unlink(temp_path)

    def test_read_empty_file(self):
        """能读取空文件"""
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as f:
            temp_path = f.name

        try:
            content = read_file(temp_path)
            assert content == ""
        finally:
            os.unlink(temp_path)


class TestChunkBySize:
    """chunk_by_size() 测试"""

    def test_chunk_normal(self):
        """能按 chunk_size / overlap 正确分块"""
        text = "0123456789" * 5  # 50 字符
        chunks = chunk_by_size(text, chunk_size=20, overlap=5)
        assert chunks == [
            "01234567890123456789",
            "56789012345678901234",
            "01234567890123456789",
        ]

    def test_chunk_exact_fit(self):
        """文本长度正好是 chunk_size 的倍数"""
        text = "0123456789" * 4  # 40 字符
        chunks = chunk_by_size(text, chunk_size=20, overlap=5)
        assert chunks == [
            "01234567890123456789",
            "56789012345678901234",
            "0123456789",
        ]

    def test_chunk_smaller_than_chunk_size(self):
        """文本小于 chunk_size 时返回单块"""
        text = "hello"
        chunks = chunk_by_size(text, chunk_size=20, overlap=5)
        assert chunks == ["hello"]

    def test_overlap_greater_equal_chunk_size_raises(self):
        """overlap >= chunk_size 时抛 ValueError"""
        text = "hello world"
        with pytest.raises(ValueError, match="参数overlap"):
            chunk_by_size(text, chunk_size=10, overlap=10)

        with pytest.raises(ValueError, match="参数overlap"):
            chunk_by_size(text, chunk_size=10, overlap=15)


class TestGetChunks:
    """get_chunks() 测试"""

    def test_filter_empty_paragraphs(self):
        """过滤空段落"""
        text = "第一段\n\n\n\n第二段\n\n"
        chunks = get_chunks(text, chunk_size=100, overlap=10)
        assert chunks == ["第一段", "第二段"]

    def test_short_paragraph_preserved(self):
        """短段落直接保留"""
        text = "短句。\n\n另一段。"
        chunks = get_chunks(text, chunk_size=100, overlap=10)
        assert chunks == ["短句。", "另一段。"]

    def test_long_paragraph_split(self):
        """超长段落调用固定大小切分"""
        text = "A" * 50 + "\n\n" + "B" * 200
        chunks = get_chunks(text, chunk_size=100, overlap=20)
        # 第一段 50 字符 < 100，直接保留
        # 第二段 200 字符 > 100，需要切分
        assert chunks[0] == "A" * 50
        assert len(chunks[1]) == 100
        assert chunks[1].startswith("B")

    def test_overlap_greater_equal_chunk_size_raises(self):
        """overlap >= chunk_size 时抛 ValueError"""
        text = "hello world"
        with pytest.raises(ValueError, match="参数overlap"):
            get_chunks(text, chunk_size=10, overlap=10)

    def test_only_whitespace_paragraphs(self):
        """只包含空白字符的段落被过滤"""
        text = "   \n\n\t\t\n\n有效内容\n\n  \n\nend"
        chunks = get_chunks(text, chunk_size=100, overlap=10)
        assert chunks == ["有效内容", "end"]


class TestAddMetadata:
    """add_metadata() 测试"""

    def test_add_metadata_fields(self):
        """添加 id / text / metadata.source / metadata.chunk_index"""
        chunks = ["第一段", "第二段", "第三段"]
        result = add_metadata("/path/to/doc.txt", chunks)

        assert len(result) == 3

        assert result[0]["id"] == "doc.txt_1"
        assert result[0]["text"] == "第一段"
        assert result[0]["metadata"]["source"] == "doc.txt"
        assert result[0]["metadata"]["chunk_index"] == 1

        assert result[1]["id"] == "doc.txt_2"
        assert result[1]["text"] == "第二段"
        assert result[1]["metadata"]["source"] == "doc.txt"
        assert result[1]["metadata"]["chunk_index"] == 2

        assert result[2]["id"] == "doc.txt_3"
        assert result[2]["text"] == "第三段"
        assert result[2]["metadata"]["source"] == "doc.txt"
        assert result[2]["metadata"]["chunk_index"] == 3

    def test_add_metadata_empty_chunks(self):
        """空 chunks 列表返回空列表"""
        result = add_metadata("/path/to/doc.txt", [])
        assert result == []

    def test_add_metadata_filename_with_path(self):
        """source 只取文件名，不含路径"""
        chunks = ["content"]
        result = add_metadata("/home/user/docs/report.pdf", chunks)
        assert result[0]["metadata"]["source"] == "report.pdf"


class TestProcessDocument:
    """process_document() 测试"""

    def test_process_document_full_pipeline(self):
        """串联 read_file → get_chunks → add_metadata"""
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as f:
            f.write("第一段内容。\n\n第二段内容，比较长一点。" + "X" * 100)
            temp_path = f.name

        try:
            results = process_document(temp_path, chunk_size=50, overlap=10)

            assert isinstance(results, list)
            assert len(results) >= 1

            for item in results:
                assert "id" in item
                assert "text" in item
                assert "metadata" in item
                assert "source" in item["metadata"]
                assert "chunk_index" in item["metadata"]
        finally:
            os.unlink(temp_path)

    def test_process_document_default_params(self):
        """使用默认 chunk_size=300 / overlap=50"""
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as f:
            f.write("测试内容。" * 50)
            temp_path = f.name

        try:
            results = process_document(temp_path)
            assert isinstance(results, list)
        finally:
            os.unlink(temp_path)

    def test_process_document_empty_file(self):
        """空文件返回空列表"""
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as f:
            temp_path = f.name

        try:
            results = process_document(temp_path)
            assert results == []
        finally:
            os.unlink(temp_path)
