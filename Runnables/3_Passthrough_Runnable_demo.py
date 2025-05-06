from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from utils.my_enums import API_KEYS, LLM_MODELS

llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
    temperature=0,
)

template = PromptTemplate(
    template="My name is {name}",
    input_variables=["name"]
)

parser = StrOutputParser()


### RunnablePassthrough does not perform any processing on the input.
# It returns the exact input as output.
chain = template | RunnablePassthrough()

result = chain.invoke({"name": "Tapoban Ray"})

print(result)