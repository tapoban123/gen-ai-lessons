from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from utils.my_enums import API_KEYS, LLM_MODELS


llm = HuggingFaceEndpoint(
    repo_id=LLM_MODELS.HF_MODEL_TINY_LLAMA,
    task="text-generation",
    huggingfacehub_api_token=API_KEYS.HF_API_KEY,
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Teach me about {topic}.", input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write me a 5 line summary on {text}.", input_variables=["text"]
)

prompt1 = template1.invoke({"topic": "black hole"})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result1.content})
result2 = model.invoke(prompt1)

print(result2.content)
