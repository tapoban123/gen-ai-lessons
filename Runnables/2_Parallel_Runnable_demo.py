from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import (
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from utils.my_enums import API_KEYS, LLM_MODELS


llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
    temperature=0,
)

template1 = PromptTemplate(
    template="Tell me a joke on {topic}.",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Provide the explanation of the {joke}.",
    input_variables=["joke"],
)

parser = StrOutputParser()


joke_gen_chain = RunnableSequence(
    template1,
    llm,
    parser,
)

joke_explain_chain = template2 | llm | parser

pass_through_chain = RunnablePassthrough()

parallel_chain = RunnableParallel(
    {
        "joke": pass_through_chain,
        "joke_explanation": joke_explain_chain,
    }
)


final_chain = joke_gen_chain | parallel_chain

result = final_chain.invoke({"topic": "AI"})

print(result)
