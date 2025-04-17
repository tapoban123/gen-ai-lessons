from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model="", dimensions=32)

# Passing our documents.
documents = [
    "Delhi is the capital of India.",
    "Kolkata is the capital of West Bengal.",
    "Paris is the capital of France.",
]

# Now it generates embeddings for each document.
result = embedding.embed_query(documents)
print(str(result))
