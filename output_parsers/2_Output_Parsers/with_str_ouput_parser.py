from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from utils.my_enums import API_KEYS, LLM_MODELS

llm = HuggingFaceEndpoint(
    repo_id=LLM_MODELS.HF_MODEL_TINY_LLAMA,
    task="text-generation",
    huggingfacehub_api_token=API_KEYS.HF_API_KEY,
)

chat_llm = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Explain me about {topic}.", input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write me a 5 line summary of {text}.", input_variables=["text"]
)

parser = StrOutputParser()

# Using OutputParser with chains
chain = template1 | chat_llm | parser | template2| chat_llm | parser

result = chain.invoke({"topic": "Celestial Bodies"})

print(result)