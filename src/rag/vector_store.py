import chromadb
from src.config import CHROMA_PATH

_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

def get_or_create_collection(collection_name: str) -> chromadb.Collection:
    return _client.get_or_create_collection(
        name = collection_name,
        metadata={"hnsw:space": "cosine"},
    )

def upsert_chunks(chunks: list[dict], embeddings: list[list[float]], collection_name: str) -> None:
    """将文档存入数据库"""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks 和 embeddings 数量不一致")
    
    if not chunks:
        return
    
    collection = get_or_create_collection(collection_name)

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    collection.upsert(
        ids = ids,
        embeddings = embeddings,
        documents = documents,
        metadatas = metadatas
    )

def query_collection(query_embedding: list[float], top_k: int, collection_name: str) -> list[dict]:
    """查询数据库，获得top_k个结果"""
    try:
        collection = _client.get_collection(
            name = collection_name,
        )

    except ValueError:
        raise ValueError(f"Collection not found: {collection_name}")

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = top_k
    )

    if not results.get("documents"):
        return []

    documents = results.get("documents")[0]
    metadatas = results.get("metadatas")[0]
    distances = results.get("distances")[0]

    retrieved: list[dict] = [] 
    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved.append(
            {
                "reference_text": doc,
                "source": meta.get('source'),
                "chunk_index": meta.get('chunk_index'),
                "distance": dist
            }
        )

    return retrieved