from langchain_nvidia_ai_endpoints import ChatNVIDIA
from utils.my_enums import API_KEYS, LLM_MODELS

client = ChatNVIDIA(
    model=LLM_MODELS.DEEPSEEK_R1_MODEL,
    api_key=API_KEYS.NVIDIA_NIM_API_KEY,
    temperature=0.6,
    top_p=0.7,
    max_tokens=4096,
)

response = client.invoke([{"role": "user", "content": "What do you know about earth?"}])
print(response.content)
