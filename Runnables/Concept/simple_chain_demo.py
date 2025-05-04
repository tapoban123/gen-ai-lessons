from .runnables_under_hood import (
    FakeLLM,
    FakePromptTemplate,
    FakeStrOutputParser,
    RunnableConnector,
)

llm = FakeLLM()

template = FakePromptTemplate(
    template="Generate a poem of {length} on {topic}.",
    input_variables=["length", "topic"],
)


parser = FakeStrOutputParser()

chain = RunnableConnector([template, llm, parser])


result = chain.invoke({"length": "long", "topic": "India"})

print(result)
