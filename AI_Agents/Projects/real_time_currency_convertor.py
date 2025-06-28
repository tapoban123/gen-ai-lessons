from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.my_enums import API_KEYS, LLM_MODELS
import requests


@tool
def get_conversion_factor(base, target) -> float:
    """Function that fetches the currency conversion factor between base currency and target currency."""
    url = f"https://v6.exchangerate-api.com/v6/{API_KEYS.EXCHANGE_RATE_API_KEY.value}/pair/{base}/{target}"
    response = requests.get(url)
    return response.json()["conversion_rate"]


result = get_conversion_factor.invoke({"base": "USD", "target": "INR"})

print(result)