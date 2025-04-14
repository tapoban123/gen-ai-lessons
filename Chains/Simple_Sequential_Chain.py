from langchain.chains.sequential import SimpleSequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from utils.my_enums import LLM_MODELS, API_KEYS
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
    temperature=0,
    max_retries=2,
)

prompt = PromptTemplate.from_template(
    "Suggest me some good names for a {type} restaurant."
)

prompt2 = PromptTemplate.from_template(
    "Suggest me some of the best dishes to be served at {type} restaurant."
)

chain1 = prompt | llm | StrOutputParser()
chain2 = prompt2 | llm | StrOutputParser()

final_chain = SimpleSequentialChain(
    chains=[chain1, chain2], input_key="indian", output_key="type"
)
# ai_answer = final_chain({"type": "indian", "cuisine": "indian"})
# print(ai_answer)
