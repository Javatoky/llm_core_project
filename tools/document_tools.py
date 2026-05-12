from errors import ToolExecutionError

from ..rag.rag_agent import retrieve


def make_search_documents_tool(collection_name: str):
    async def search_documents(query: str) -> dict:
        query = query.strip()
        if not query:
            raise ToolExecutionError("查询语句不能为空")

        try:
            results = await retrieve(query, collection_name)
        except Exception as e:
            raise ToolExecutionError(f"查找出错：{e}")
        if not results:
            raise ToolExecutionError(f"知识库中没有找到相关资料")
        return results
    
    return search_documents