# %pip install -qU langchain-google-genai

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,   
)

messages = [
    (
        "system",
        "You are a helpful assistent that help students of school and college to simplify the activity of studying."
    ),
    ("human", "I want to learn to build mobile apps. How should I start?")
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)