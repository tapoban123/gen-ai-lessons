from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.my_enums import API_KEYS, LLM_MODELS
from langchain_core.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
)


parser = JsonOutputParser()

template = PromptTemplate(
    template="Generate me name, city and job title of 5 fictional persons. \n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# prompt = template.format()

# result = llm.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | llm | parser

result = chain.invoke({})

print(result)
