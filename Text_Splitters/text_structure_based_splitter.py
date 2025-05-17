from langchain.text_splitter import RecursiveCharacterTextSplitter
from .texts import medium_text

splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=0,
)

chunks = splitter.split_text(medium_text)

print(chunks)
