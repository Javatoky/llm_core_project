from src.adapters.model_adapter import LLMBackend
from src.config import RAG_CONFIG
from src.rag.document_pipeline import process_document
from src.rag.vector_store import query_collection, upsert_chunks

RAG_SYSTEM_PROMPT = """
你是一个基于本地知识库回答问题的助手。

规则：
- 只能基于提供的 Context 回答。
- 如果 Context 中没有答案，回答“根据现有资料无法回答此问题。”
- 回答中必须标注来源，格式为：source: 文件名, chunk_index: 编号。
""".strip()

async def build_vector_store(backend: LLMBackend, filepaths: list[str], collection_name: str) -> None:
    """文档存储流"""
    chunks: list[dict] = []

    for filepath in filepaths:
        doc_chunks = process_document(filepath)
        chunks.extend(doc_chunks)

    if not chunks:
        return
    
    texts = [chunk["text"] for chunk in chunks]
    embeddings = await backend.embed(texts)

    upsert_chunks(chunks, embeddings, collection_name)

async def retrieve(backend: LLMBackend, query: str, collection_name: str, top_k: int = RAG_CONFIG["top_k"], distance_threshold: float = RAG_CONFIG["distance_threshold"]) -> list[dict]:
    """信息检索"""
    query_embedding = await backend.embed([query])
    query_embedding = query_embedding[0]
    
    results = query_collection(
        query_embedding = query_embedding,
        top_k = top_k,
        collection_name = collection_name
    )
 
    filtered_results = [
        result
        for result in results
        if result["distance"] <= distance_threshold
    ]

    return filtered_results

def format_context(texts: list[dict]) -> str:
    return "\n\n".join(
        f"[source]: {text['source']}, [chunk_index]: {text['chunk_index']}\n"
        f"[text]: {text['reference_text']}"
        for text in texts
    )

async def answer_with_rag(
    backend: LLMBackend,
    user_input: str,
    collection_name: str,
    messages: list[dict]
) -> str:
    """问答过程"""
    texts = await retrieve(backend, user_input, collection_name)
    if not texts:
        reply = f"根据现有资料无法回答此问题。"
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": reply})
        return reply
    
    rag_messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user", 
            "content":
                f"【参考文本信息】\n{format_context(texts)}\n\n【用户问题】\n{user_input}"
        }
    ]
    
    reply = await backend.chat(rag_messages)

    messages.append(
        {"role": "user", "content": user_input}
    )
    messages.append(
        {"role": "assistant", "content": reply}
    )

    return reply