from pathlib import Path


def read_file(filepath: str) -> str:
    """读取文本"""
    path = Path(filepath)
    with open(path, "r", encoding = "utf-8") as f:
        text = f.read()
    
    return text

def chunk_by_size(para: str, chunk_size: int, overlap: int) -> list[str]:
    """按大小分块"""
    if chunk_size <= overlap:
        raise ValueError("参数overlap必须小于chunk_size！")
    
    start = 0
    chunks: list[str] = []
    length = len(para)
    while True:
        if (end:=start + chunk_size) >= length:
            chunks.append(para[start: length])
            break

        chunks.append(para[start: end])

        start = end - overlap
    
    return chunks

def get_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """分块处理"""
    if chunk_size <= overlap:
        raise ValueError("参数overlap必须小于chunk_size！")
    
    chunks: list[str] = []
    while True:
        if not (text:=text.strip()):
            break
        para, _, text = text.partition("\n\n")
        if not para.strip():
            continue

        if len(para) < chunk_size:
            chunks.append(para)
        else:
            chunks.extend(chunk_by_size(para, chunk_size, overlap))

    return chunks

def add_metadata(filepath: str, chunks: list[str]) -> list[dict]:
    """元数据处理"""
    source = Path(filepath).name

    return [
        {
            "id": f"{source}_{i}",
            "text": chunk,
            "metadata": {
                "source": source,
                "chunk_index": i
            }
        }
        for i, chunk in enumerate(chunks, 1)
    ]

def process_document(filepath: str, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """处理单个文档"""
    text = read_file(filepath)
    chunks = get_chunks(text)
    return add_metadata(filepath, chunks)