from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.my_enums import API_KEYS, LLM_MODELS
from pydantic import BaseModel, Field
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


class CurrencyModel(BaseModel):
    base: str = Field(description="Currency code of base currency.")
    target: str = Field(description="Currency code of target currency.")
    amount: float = Field(description="Amount to be converted to a different currency.")


llm = ChatGoogleGenerativeAI(
    model=LLM_MODELS.GEMINI_MODEL.value,
    api_key=API_KEYS.GEMINI_API_KEY.value,
)

currency_output_parser = PydanticOutputParser(pydantic_object=CurrencyModel)

currency_prompt = PromptTemplate(
    template="""Get the currency codes and amount to be converted from the following query:
    {query}
    \n{format_instructions}
    """,
    input_variables=["query"],
    partial_variables={"format_instructions": currency_output_parser.get_format_instructions()}
)

user_query = input("Enter a query: ")
currency_chain = currency_prompt | llm | currency_output_parser
currency_details = currency_chain.invoke({"query": user_query})

# query = input("Enter a query: ")
# chat_history = [HumanMessage(f"Get the currency codes and conversion factor of the given query and {query}")]

query = f"""Get the conversion factor between {currency_details.base} and {currency_details.target} 
and convert {currency_details.amount} {currency_details.base} to {currency_details.target}?"""
chat_history = [HumanMessage(query)]

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

ai_message = llm_with_tools.invoke(chat_history)
chat_history.append(ai_message)

for msg in ai_message.tool_calls:
    conversion_factor: float

    if msg["name"] == "get_conversion_factor":
        tool_msg = get_conversion_factor.invoke(msg)
        conversion_factor = tool_msg.content
        chat_history.append(tool_msg)
    elif msg["name"] == "convert":
        msg["args"]["conversion_rate"] = conversion_factor
        tool_message = convert.invoke(msg)
        chat_history.append(tool_message)
#
# print(chat_history)
result = llm_with_tools.invoke(chat_history)
print(result.content)
