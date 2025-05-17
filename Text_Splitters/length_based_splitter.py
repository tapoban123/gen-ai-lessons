from langchain.text_splitter import CharacterTextSplitter
from .texts import medium_text


splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=5, separator="")

# If we split text from a document then every chunk will be a Document object.
result = splitter.split_text(text=medium_text)

print(result)
print(result[0])
