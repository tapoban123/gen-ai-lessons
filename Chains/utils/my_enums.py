from enum import Enum
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API = os.environ.get("GEMINI_API_KEY")
GOOGLE_SERP_API = os.environ.get("GOOGLE_SERP_API")


class API_KEYS(Enum):
    GEMINI_API_KEY = GEMINI_API
    SERP_API_KEY = GOOGLE_SERP_API


class LLM_MODELS(Enum):
    GEMINI_MODEL = "gemini-2.0-flash"
