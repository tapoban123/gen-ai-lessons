from enum import Enum
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API = os.environ.get("GEMINI_API_KEY")
GOOGLE_SERP_API = os.environ.get("GOOGLE_SERP_API")
HF_API = os.environ.get("HUGGING_FACE_API_TOKEN")


class API_KEYS(Enum):
    GEMINI_API_KEY = GEMINI_API
    SERP_API_KEY = GOOGLE_SERP_API
    HF_API_KEY = HF_API


class LLM_MODELS(Enum):
    GEMINI_MODEL = "gemini-2.0-flash"
    HF_MODEL_TINY_LLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    HF_EMBEDDING_MODEL_SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"
    GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
