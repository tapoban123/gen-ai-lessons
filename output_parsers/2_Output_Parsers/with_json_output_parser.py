from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from utils.my_enums import API_KEYS, LLM_MODELS
from langchain_core.prompts import PromptTemplate

llm = HuggingFaceEndpoint(
    repo_id=LLM_MODELS.HF_MODEL_TINY_LLAMA,
    task="text-generation",
    huggingfacehub_api_token=API_KEYS.HF_API_KEY,
)

chat_llm = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Generate me name, city and job title of 5 fictional persons. \n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

prompt = template.format()

result = chat_llm.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
