from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import load_prompt
from utils.my_enums import API_KEYS, LLM_MODELS

prompt_template = load_prompt("Prompt_Templating/research_prompt.json")

llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY, model=LLM_MODELS.GEMINI_MODEL
)

prompt = prompt_template.invoke({"title": "Attention is All You Need"})

result = llm.invoke(prompt)

print(result.content)
