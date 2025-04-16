from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-3.5-turbo-instruct")

response = llm.invoke("What is the capital of India?")

print(response.content)
