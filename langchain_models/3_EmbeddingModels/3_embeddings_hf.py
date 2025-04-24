from langchain_huggingface import HuggingFaceEndpointEmbeddings
from utils.my_enums import API_KEYS, LLM_MODELS


embedding = HuggingFaceEndpointEmbeddings(
    model=LLM_MODELS.HF_EMBEDDING_MODEL_SENTENCE_TRANSFORMER,
    task="feature-extraction",
    huggingfacehub_api_token=API_KEYS.HF_API_KEY,
)

# text = "Delhi is the capital of India."

# Passing out documents.
documents = [
    "Delhi is the capital of India.",
    "Kolkata is the capital of West Bengal.",
    "Paris is the capital of France.",
]

doc_embeddings = []

for doc in documents:
    vector = embedding.embed_query(doc)
    doc_embeddings.append(str(vector))

print(doc_embeddings)
