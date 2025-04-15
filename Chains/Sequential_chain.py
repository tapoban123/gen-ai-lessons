from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.sequential import SequentialChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from utils.my_enums import API_KEYS, LLM_MODELS

llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY.value,
    model=LLM_MODELS.GEMINI_MODEL.value,
    temperature=0,
    max_retries=2,
)

prompt = PromptTemplate.from_template(
    "I want to start a company that manufactures {product}. Please tell me a good name for my company."
)

prompt2 = PromptTemplate.from_template(
    "Tell me the strategy of building a successful business in {industry}."
)

chain1 = prompt | llm | StrOutputParser()
chain2 = prompt2 | llm | StrOutputParser()

final_chain = chain1 | chain2
ai_answer = final_chain.invoke({"product": "Electronic Gadgets", "industry": "IOT"})
print(ai_answer)
