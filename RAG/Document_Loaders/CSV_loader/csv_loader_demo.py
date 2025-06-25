from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="Document_Loaders/CSV_loader/RS_Session_258_AU_1432_1.csv")

docs = loader.load()

print(f"Number of rows/documents: {len(docs)}", end="\n\n")

print(docs[0].metadata)
print(docs[0].page_content)

print("\n\nLazy Loading Documents:")

iterable_docs = loader.lazy_load()

for doc in iterable_docs:
    print(doc.page_content, end="\n\n")