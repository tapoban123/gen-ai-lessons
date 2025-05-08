from langchain_community.document_loaders import TextLoader


loader = TextLoader(
    "Document_Loaders/text_loader/quantum_computing.txt",
    encoding="utf-8",
)

docs = loader.load()

print(len(docs))

print(type(docs)) # Document object

print(docs[0].metadata)
