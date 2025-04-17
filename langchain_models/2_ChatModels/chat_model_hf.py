from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from utils.my_enums import API_KEYS, LLM_MODELS

llm = HuggingFaceEndpoint(
    repo_id=LLM_MODELS.HF_MODEL_TINY_LLAMA,
    task="text-generation",
    huggingfacehub_api_token=API_KEYS.HF_API_KEY,
)

model = ChatHuggingFace(llm=llm)

ans = model.invoke("What is the capital of India?")

print(ans.content)
