from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.my_enums import API_KEYS, LLM_MODELS

@tool
def multiply(a: int, b: int) -> int:
    """Method that takes input of two numbers and calculates the product of them."""
    return a * b

llm = ChatGoogleGenerativeAI(
    model=LLM_MODELS.GEMINI_MODEL.value,
    api_key=API_KEYS.GEMINI_API_KEY.value,
)


### Binding tools with LLM
llm_with_tools = llm.bind_tools([multiply])

query = "Can you multiply 5 and 6?"
result = llm_with_tools.invoke(query)

print(result, end="\n\n")

## Calling tools
tool_message = result.tool_calls
tool_result = multiply.invoke(tool_message[0])

print(tool_result)

