from enum import Enum
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API = os.environ.get("GEMINI_API_KEY")
GOOGLE_SERP_API = os.environ.get("GOOGLE_SERP_API")
HF_API = os.environ.get("HUGGING_FACE_API_TOKEN")
OPEN_ROUTER_API = os.environ.get("OPEN_ROUTER_API_KEY")
NVIDIA_NIM_API = os.environ.get("NVIDIA_NIM_API_KEY")


class BASE_URLS(Enum):
    OPEN_ROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class API_KEYS(Enum):
    GEMINI_API_KEY = GEMINI_API
    SERP_API_KEY = GOOGLE_SERP_API
    HF_API_KEY = HF_API
    OPEN_ROUTER_API_KEY = OPEN_ROUTER_API
    NVIDIA_NIM_API_KEY = NVIDIA_NIM_API


class LLM_MODELS(Enum):
    GEMINI_MODEL = "gemini-2.0-flash"
    HF_MODEL_TINY_LLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    HF_MODEL_MS_BITNET = "microsoft/bitnet-b1.58-2B-4T"
    HF_MODEL_MINICHAT = "GeneZC/MiniChat-2-3B"
    HF_EMBEDDING_MODEL_SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"
    GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
    NVIDIA_NEMOTRON_ULTRA_MODEL = "nvidia/llama-3.1-nemotron-ultra-253b-v1:free"
    DEEPSEEK_V3_BASE_MODEL = "deepseek/deepseek-v3-base:free"
    DEEPSEEK_R1_MODEL = "deepseek-ai/deepseek-r1"
    QWEN_VL_MODEL = "qwen/qwen2.5-vl-32b-instruct:free"
