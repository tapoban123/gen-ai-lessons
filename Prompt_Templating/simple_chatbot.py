from langchain_google_genai import ChatGoogleGenerativeAI
from utils.my_enums import API_KEYS, LLM_MODELS

llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
)


while True:
    user_input = input("User: ")
    if (user_input.strip().lower()) in ["end", "exit"]:
        break
    llm_output = llm.invoke(user_input)
    print(f"Gemini: {llm_output.content}\n")
