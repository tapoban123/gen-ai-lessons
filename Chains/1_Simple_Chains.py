from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.my_enums import API_KEYS, LLM_MODELS


llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY, model=LLM_MODELS.GEMINI_MODEL
)


template = PromptTemplate(
    template="Explain me about {topic}.",
    input_variables=["topic"],
)

parser = StrOutputParser()

chain = template | llm | parser

result = chain.invoke({"topic": "Moon"})

# print(result)

## Visualise the flow of the chain.
chain.get_graph().print_ascii()