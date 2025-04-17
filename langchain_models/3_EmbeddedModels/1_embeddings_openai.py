from langchain_openai import OpenAIEmbeddings

# Larger the dimansion, more context is captured.
embedding = OpenAIEmbeddings(model="", dimensions=32)

# (method) def embed_query(text: str) -> list[float]
result = embedding.embed_query("Delhi is the capital of India.")

print(str(result))