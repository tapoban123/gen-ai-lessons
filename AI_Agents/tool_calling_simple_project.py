from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from utils.my_enums import API_KEYS, LLM_MODELS

@tool
def multiply(a: int, b: int):
    """Method to take two numbers as input and calculate their product."""
    return a * b


llm = ChatGoogleGenerativeAI(
    model=LLM_MODELS.GEMINI_MODEL.value,
    api_key=API_KEYS.GEMINI_API_KEY.value,
)

# query = "Can you return the product of 5 and 6?"
query = input("Enter a query: ")

llm_with_tools = llm.bind_tools([multiply])
chat_history = [HumanMessage(query)]

ai_message = llm_with_tools.invoke(chat_history)
chat_history.append(ai_message)

tool_message = multiply.invoke(ai_message.tool_calls[0])
chat_history.append(tool_message)

result =  llm_with_tools.invoke(chat_history)

print(result.content)



