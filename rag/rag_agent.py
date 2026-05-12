import copy
import json

from ..adapters.model_adapter import LLMBackend
from ..config import RAG_CONFIG
from .document_pipeline import process_document
from .vector_store import query_collection, upsert_chunks


async def build_vector_store(backend: LLMBackend, filepaths: list[str], collection_name: str) -> None:
    """文档存储流"""
    chunks: list[dict] = []
    embeddings: list[list[float]] = []

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
        qeury_embedding = query_embedding,
        top_k = top_k,
        collection_name = collection_name
    )
 
    filtered_results = [
        result
        for result in results
        if result["distance"] <= distance_threshold
    ]
    if not filtered_results:
        return []

    return filtered_results

async def answer_with_rag(
    backend: LLMBackend,
    user_input: str,
    collection_name: str,
    messages: list[dict]
) -> str:
    """问答过程"""
    texts = await retrieve(backend, user_input, collection_name)
    if not texts:
        reply = f"根据现有资料无法回答此问题"
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": reply})
        return reply
    
    temp_messages = copy.deepcopy(messages)
    temp_messages.append(
        {
            "role": "user", 
            "content":
                f"【参考文本信息】\n"
                f"{json.dumps(texts, ensure_ascii=False)}\n\n"
                f"【用户问题】{user_input}"
        }
    )
    reply = await backend.chat(temp_messages)

    messages.append(
        {"role": "user", "content": user_input}
    )
    messages.append(
        {"role": "assistant", "content": reply}
    )

    return reply