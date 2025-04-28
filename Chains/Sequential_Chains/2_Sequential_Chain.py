from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.my_enums import API_KEYS, LLM_MODELS


llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
    temperature=0,
)

template = PromptTemplate(
    template="Provide me descriptive report on {topic}.",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Summarise the following {text}.",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = template | llm | parser | template2 | llm | parser

result = chain.invoke({"topic": "Many Worlds Interpretation of Quantum Mechanics."})

print(result)

chain.get_graph().print_ascii()