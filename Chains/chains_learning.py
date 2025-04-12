from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from utils.my_enums import API_KEYS, LLM_MODELS

prompt = PromptTemplate.from_template(
    "What is a good name for a company that manufactures {product}?"
)

llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
    temperature=0,
    max_retries=2,
)

# Connecting llm and prompt.
chain = prompt | llm | StrOutputParser()

ai_answer = chain.invoke("Toys")

print(ai_answer)
