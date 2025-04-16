from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("What is the capital of India?")

print(result)