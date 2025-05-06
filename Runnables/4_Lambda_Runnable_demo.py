from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from utils.my_enums import API_KEYS, LLM_MODELS


llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
    temperature=0.5,
)

template = PromptTemplate(
    template="Generate me a summary on {topic}.",
    input_variables=["topic"],
)

parser = StrOutputParser()

def words_count(text: str):
    return len(text.split())


summary_chain = template | llm | parser

### RunnableLambda is used to invoke a user-defined function inside a chain.
lambda_runnable = RunnableLambda(words_count)

parallel_chain = RunnableParallel(
    {
        "summary": RunnablePassthrough(),
        "words_count": lambda_runnable,
    }
)

final_chain = summary_chain | parallel_chain

result = final_chain.invoke({"topic": "Randomness in Quantum Mechanics"})

print(result)