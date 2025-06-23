from langchain.retrievers import ArxivRetriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from utils.my_enums import API_KEYS, LLM_MODELS


def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
)

prompt = PromptTemplate(
    template="""Answer the question based only on the context provided.

Context: {context}

Question: {question}""",
    input_variables=["context", "question"],
)


query = input("Enter a query: ")

retriever = ArxivRetriever(get_full_documents=True)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke(query)

print(result)