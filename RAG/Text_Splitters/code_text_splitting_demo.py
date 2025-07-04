from langchain.text_splitter import RecursiveCharacterTextSplitter,Language


code = """
def hello():    return "Hello World"
"""

# An example of Document-structure based text splitting.
splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=100,
    chunk_overlap=0,
    language=Language.PYTHON,
)

chunks = splitter.split_text(code)

print(chunks)