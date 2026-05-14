"""tools/registry.py 测试模块

测试 src/tools/registry.py 中的核心函数。
"""

from src.tools.registry import build_tool_registry, build_tool_schemas

# ==================== build_tool_schemas 测试 ====================


def test_build_tool_schemas_without_rag():
    """测试 build_tool_schemas(include_rag=False) 不包含 search_documents"""
    schemas = build_tool_schemas(include_rag=False)

    schema_names = [schema["function"]["name"] for schema in schemas]

    assert "search_documents" not in schema_names
    assert "get_weather" in schema_names
    assert "calculate" in schema_names


def test_build_tool_schemas_with_rag():
    """测试 build_tool_schemas(include_rag=True) 包含 search_documents"""
    schemas = build_tool_schemas(include_rag=True)

    schema_names = [schema["function"]["name"] for schema in schemas]

    assert "search_documents" in schema_names
    assert "get_weather" in schema_names
    assert "calculate" in schema_names


# ==================== build_tool_registry 测试 ====================


def test_build_tool_registry_without_search_documents():
    """测试 build_tool_registry(search_documents=None) 不包含 search_documents"""
    registry = build_tool_registry(search_documents=None)

    assert "search_documents" not in registry
    assert "get_weather" in registry
    assert "calculate" in registry


def test_build_tool_registry_with_search_documents():
    """测试 build_tool_registry(search_documents=fake_tool) 包含 search_documents"""
    async def fake_search_documents(query: str) -> str:
        return f"搜索结果：{query}"

    registry = build_tool_registry(search_documents=fake_search_documents)

    assert "search_documents" in registry
    assert registry["search_documents"][0] is fake_search_documents
    assert registry["search_documents"][1] == ["query"]


# ==================== schema 与 registry 一致性测试 ====================


def test_schema_names_match_registry_keys():
    """测试 schema 中暴露的工具名必须都能在 registry 中找到"""
    schemas = build_tool_schemas(include_rag=False)
    registry = build_tool_registry(search_documents=None)

    schema_names = {schema["function"]["name"] for schema in schemas}
    registry_keys = set(registry.keys())

    assert schema_names == registry_keys


def test_schema_names_match_registry_keys_with_rag():
    """测试包含 RAG 时，schema 中暴露的工具名必须都能在 registry 中找到"""
    async def fake_search_documents(query: str) -> str:
        return f"搜索结果：{query}"

    schemas = build_tool_schemas(include_rag=True)
    registry = build_tool_registry(search_documents=fake_search_documents)

    schema_names = {schema["function"]["name"] for schema in schemas}
    registry_keys = set(registry.keys())

    assert schema_names == registry_keys
