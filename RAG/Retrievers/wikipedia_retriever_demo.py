from langchain.retrievers import WikipediaRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from utils.my_enums import LLM_MODELS, API_KEYS


def formatDocs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


llm = ChatGoogleGenerativeAI(
    model=LLM_MODELS.GEMINI_MODEL,
    api_key=API_KEYS.GEMINI_API_KEY,
)

query = input("Enter your query: ")

prompt = PromptTemplate(
    template="""You are a helpful assistent.
    Answer the question based only on the context provided.

Context: {context}

Question: {question}""",
    input_variables=["context", "question"],
)

retriever = WikipediaRetriever()


chain = (
    {"context": retriever | formatDocs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke(query)

print(result)

