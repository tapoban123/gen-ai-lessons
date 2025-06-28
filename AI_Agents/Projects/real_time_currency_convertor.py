from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.my_enums import API_KEYS, LLM_MODELS
from typing import Annotated
import requests


@tool
def get_conversion_factor(base: str, target: str) -> float:
    """Function that fetches the currency conversion factor between base currency and target currency."""
    url = f"https://v6.exchangerate-api.com/v6/{API_KEYS.EXCHANGE_RATE_API_KEY.value}/pair/{base}/{target}"
    response = requests.get(url)
    return response.json()["conversion_rate"]


@tool
def convert(base: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """this function converts the base currency to target currency value."""
    return base * conversion_rate


query = "Get the conversion factor between USD and INR and convert 20 dollars to INR?"
chat_history = [HumanMessage(query)]

llm = ChatGoogleGenerativeAI(
    model=LLM_MODELS.GEMINI_MODEL.value,
    api_key=API_KEYS.GEMINI_API_KEY.value,
)

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

ai_message = llm_with_tools.invoke(chat_history)
chat_history.append(ai_message)

# query2 = "based on the conversion factor can you convert 20 dollars to INR?"
# chat_history.append(HumanMessage(query2))

# llm_with_tools.invoke(chat_history)

# result = get_conversion_factor.invoke({"base": "USD", "target": "INR"})
# result = convert.invoke({"base": "22", "conversion_rate": "85.5582"})

print(chat_history)
print(ai_message.tool_calls)

for msg in ai_message.tool_calls:
    tool_message: ToolMessage

    if msg["name"] == "get_conversion_factor":
        tool_message = get_conversion_factor.invoke(msg)
    elif msg["name"] == "convert":
        tool_message = convert.invoke(msg)

    chat_history.append(tool_message)


result = llm_with_tools.invoke(chat_history)
print(result.content)
