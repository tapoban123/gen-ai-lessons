from .runnables_under_hood import (
    FakeLLM,
    FakePromptTemplate,
    FakeStrOutputParser,
    RunnableConnector,
)


llm = FakeLLM()

template1 = FakePromptTemplate(
    template="Tell me a joke on {topic}.",
    input_variables=["topic"],
)


template2 = FakePromptTemplate(
    template="Explain me the joke {response}.",
    input_variables=["response"],
)


parser = FakeStrOutputParser()

chain1 = RunnableConnector(
    [
        template1,
        llm,
    ]
)

chain2 = RunnableConnector(
    [
        template2,
        llm,
        parser,
    ]
)

final_chain = RunnableConnector(
    [
        chain1,
        chain2,
    ]
)

result = final_chain.invoke({"topic": "Cartoon"})

print(result)
