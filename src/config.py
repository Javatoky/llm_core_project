# config.py
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_ALY_API_KEY = os.getenv("QWEN_ALY_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1")
QWEN_ALY_BASE_URL = os.getenv("QWEN_ALY_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-plus")
QWEN_EMBED_MODEL = os.getenv("QWEN_EMBED_MODEL", "text-embedding-v1")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.5")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-5")
DS_API_KEY = os.getenv("DS_API_KEY")
DS_BASE_URL = os.getenv("DS_BASE_URL")
DS_MODEL = os.getenv("DS_MODEL")

CHROMA_PATH = Path(__file__).resolve().parents[1] / "chromadb"

RAG_CONFIG = {
    "top_k": 3,
    "distance_threshold": 0.7
}
