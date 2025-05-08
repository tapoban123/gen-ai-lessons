from langchain_community.document_loaders import WebBaseLoader


urls = ["https://patrol.leancode.co/documentation/write-your-first-test"]

loader = WebBaseLoader(urls)

docs = loader.load()

print(docs[0].metadata)