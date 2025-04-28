from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.my_enums import API_KEYS, LLM_MODELS, BASE_URLS
from langchain.schema.runnable import RunnableParallel
from langchain_core.output_parsers import StrOutputParser


llm1 = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
    temperature=0,
)

llm2 = ChatOpenAI(
    base_url=BASE_URLS.OPEN_ROUTER_BASE_URL,
    api_key=API_KEYS.OPEN_ROUTER_API_KEY,
    model=LLM_MODELS.NVIDIA_NEMOTRON_ULTRA_MODEL.value,
)


template1 = PromptTemplate(
    template="Generate well-simplified notes on {content}.",
    input_variables=["content"],
)

template2 = PromptTemplate(
    template="Generate quiz questions on {content}",
    input_variables=["content"],
)

template3 = PromptTemplate(
    template="Merge the following into one single response.\nNotes: {notes}\nQuiz:{quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": template1 | llm1 | parser,
        "quiz": template2 | llm2 | parser,
    }
)

merge_chain = template3 | llm1 | parser

chain_of_chains = parallel_chain | merge_chain

with open("Chains/Parallel_Chains/content.txt", mode="r") as f_read:
    content = f_read.read()
    result = chain_of_chains.invoke({"content": content})
    
    with open("Chains/Parallel_Chains/notes_and_quiz.md", mode="w") as f_write:
        f_write.write(result)
        

chain_of_chains.get_graph().print_ascii()   