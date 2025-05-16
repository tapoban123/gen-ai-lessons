from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loader = DirectoryLoader(
    path="Document_Loaders/directory_loader/papers",
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)

docs = loader.load()

total_document_objects = len(docs) # sum of pages from all pdfs.

print(f"Number of Document Objects: {total_document_objects}")

print(docs[11].page_content)
print(docs[11].metadata)