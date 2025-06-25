from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path="Document_Loaders/pdf_loader/Attention_is_all_you_need.pdf",
)

docs = loader.load()

# In case of PDFs, each page one Document Object.
print(docs[1].page_content)


