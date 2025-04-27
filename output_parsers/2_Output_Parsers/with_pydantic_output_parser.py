from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field, BaseModel
from utils.my_enums import API_KEYS, LLM_MODELS


llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
    temperature=0,
)


class Person(BaseModel):
    name: str = Field(description="Name of the person.")
    age: int = Field(gt=18, description="Age of the person.")
    job_title: str = Field(description="Job title of the person.")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate me name, age and job_title of a {region} persons. \n{format_instructions}",
    input_variables=["region"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# prompt = template.invoke({"region": "African"})

# result = llm.invoke(prompt)

# final_output = parser.parse(result.content)


chain = template | llm | parser

result = chain.invoke({"region": "Tribal"})

print(result.model_dump())
