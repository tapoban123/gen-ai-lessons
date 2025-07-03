from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from utils.my_enums import LLM_MODELS, API_KEYS

# 1. Creating the tools
search_tool = DuckDuckGoSearchRun()

# 2. Initialising the LLM
llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY.value,
    model=LLM_MODELS.GEMINI_MODEL.value,
)

# 3. Pulling the standard ReAct Agent prompt from Langchain Hub
prompt = hub.pull("hwchase17/react")

# 4. Creating a ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt,
)

# 5. Wrap the agent with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
)

# 6. Invoke input
response = agent_executor.invoke({"input":"What is the population of the capital of India?"})
# response = search_tool.invoke("What is the population of India?")
print(response)