from langchain_openai import ChatOpenAI
from utils.my_enums import API_KEYS, LLM_MODELS, BASE_URLS

llm = ChatOpenAI(
    base_url=BASE_URLS.OPEN_ROUTER_BASE_URL,
    api_key=API_KEYS.OPEN_ROUTER_API_KEY.value,
    model=LLM_MODELS.QWEN_VL_MODEL.value,
)

prompt = "Tell me about celestial bodies."

result = llm.invoke(prompt)

print(result.content)
