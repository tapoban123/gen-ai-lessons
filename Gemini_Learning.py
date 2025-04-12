# %pip install langchain-google-genai

from langchain_google_genai import ChatGoogleGenerativeAI
from Chains.utils.my_enums import API_KEYS, LLM_MODELS


llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

messages = [
    (
        "system",
        "You are a helpful assistent that help students of school and college to simplify the activity of studying.",
    ),
    ("human", "I want to learn to build mobile apps. How should I start?"),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)
