from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.my_enums import API_KEYS, LLM_MODELS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = GoogleGenerativeAIEmbeddings(
    google_api_key=API_KEYS.GEMINI_API_KEY.value,
    model=LLM_MODELS.GEMINI_EMBEDDING_MODEL.value,
    task_type="SEMANTIC_SIMILARITY",
)

documents = [
    "Delhi is the capital of India.",
    "Kolkata is the capital of West Bengal.",
    "Paris is the capital of France.",
]

query = "Tell me about Delhi."

docs_vector = embedding.embed_documents(documents)
query_vector = embedding.embed_query(query)

# Both parameters take 2D lists
similarity_scores = cosine_similarity(
    [query_vector],
    docs_vector,
)[0]

float64_to_float_list = [vector.item() for vector in similarity_scores]

scores = list(enumerate(float64_to_float_list))

sorted_scores = sorted(scores, key=lambda x: x[1])

most_similar_index, most_similar_score = sorted_scores[-1]
print(query)
print(documents[most_similar_index])
