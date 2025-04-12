

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["country"],
    template="What is the capital of {country}?"
)

prompt.format(country="India")