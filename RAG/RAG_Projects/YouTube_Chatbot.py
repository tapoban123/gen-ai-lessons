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
video_url = "https://youtu.be/EzYaFF7ahKw?si=xnIcMP4-etZdDct1"
video_id = extract.video_id(url=video_url)
transcript = get_video_transcript(video_id=video_id)

# Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])


# Indexing (Embedding Generation & Storing in Vector Stores)
embeddings_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODELS.GEMINI_EMBEDDING_MODEL2.value,
    google_api_key=API_KEYS.GEMINI_API_KEY,
)

# embeddings = embeddings_model.embed_documents(chunks)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    
)
