# Used to maintain conversation history between human and ai.
from langchain_core.prompts import ChatPromptTemplate

# Presently, does not work with Messsage classes like HumanMessage, etc.
prompt_template = ChatPromptTemplate(
    [
        ("system", "You are an expert at {domain}."),
        ("human", "Explain me the concept of {topic}."),
    ]
)

prompt = prompt_template.invoke(
    {
        "domain": "Physics",
        "topic": "Free will",
    }
)

print(prompt)
