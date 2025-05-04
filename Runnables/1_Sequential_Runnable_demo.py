from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from utils.my_enums import API_KEYS, LLM_MODELS


llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
    temperature=0,
)

template = PromptTemplate(
    template="Generate a short and simple explanation on {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()


chain = RunnableSequence(template, llm, parser)

result = chain.invoke({"topic": "Quantum Computing."})

print(result)
