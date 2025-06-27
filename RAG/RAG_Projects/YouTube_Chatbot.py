from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from pytube import extract
from utils.my_enums import API_KEYS, LLM_MODELS, EMBEDDING_MODELS


# Indexing (Document Ingestion)
def get_video_transcript(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            languages=["en", "hi", "bn"],
            video_id=video_id,
        )
        formatted_transcript = " ".join(chunk["text"] for chunk in transcript)
        return formatted_transcript

    except TranscriptsDisabled:
        return "Transcripts disabled for this video."


# video_url = input("Enter YouTube video URL: ")
video_url = "https://youtu.be/r-iLBNaCTDk?si=xSfRuYUxVq3lzHot"
video_id = extract.video_id(url=video_url)
transcript = get_video_transcript(video_id=video_id)

# Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# Indexing (Embedding Generation & Storing in Vector Stores)
embeddings_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODELS.GEMINI_EMBEDDING_MODEL2.value,
    google_api_key=API_KEYS.GEMINI_API_KEY.value,
)

# embeddings = embeddings_model.embed_documents(chunks)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
)

### Setting up the Retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# search_type (Optional[str]): Defines the type of search that
# the Retriever should perform.
# Can be "similarity" (default), "mmr", or "similarity_score_threshold".


## Setting up the LLM
llm = ChatGoogleGenerativeAI(
    model=LLM_MODELS.GEMINI_MODEL,
    api_key=API_KEYS.GEMINI_API_KEY,
)
prompt = PromptTemplate(
    template="""You are a helpful assistant.
    Answer only from the provided transcript context.
    If the transcript context is insufficient, just say you don't know.
    
    {context}
    Question: {question}
    """,
    input_variables=["context", "question"],
)

question = "What are Github actions?"

retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})

## Generation
result = llm.invoke(final_prompt)
print(result.content)
