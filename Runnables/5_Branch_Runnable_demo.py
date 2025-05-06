from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
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

template2 = PromptTemplate(
    template="Compress the {text} within 300 words.", input_variables=["text"]
)

parser = StrOutputParser()

summary_chain = template | llm | parser


def count_words(text: str):
    return len(text.split())

conditional_chain = RunnableBranch(
    (lambda x: count_words(x) > 300, template2 | llm | parser),
    RunnablePassthrough(),
)

summary_with_condition_chain = summary_chain | conditional_chain | parser
count_words_runnable = RunnableLambda(count_words)


summary_words_count_chain = RunnableParallel(
    {
        "summary": RunnablePassthrough(),
        "words_count": count_words_runnable,
    }
)

final_chain = summary_with_condition_chain | summary_words_count_chain

result = final_chain.invoke({"topic": "Quantum Computing"})

print(result)