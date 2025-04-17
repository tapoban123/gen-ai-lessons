# Running HuggingFace models locally on your system.

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os


# Store the model locally on D drive instead of C drive.
os.environ["HF_HOME"] = "D://hf_model"


llm = HuggingFacePipeline.from_model_id(
    model_id="",
    task="",
)

chat = ChatHuggingFace(llm)

ans = chat.invoke("What is the capital of India?")

print(ans.content)