# from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, load_tools
from langchain.agents import AgentType
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_SERP_API_KEY = os.environ.get("GOOGLE_SERP_API")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
print(GOOGLE_SERP_API_KEY)
llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

tools = load_tools(tool_names=["serpapi"], serpapi_api_key=GOOGLE_SERP_API_KEY, llm=llm)

self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
self_ask_with_search.run("What are some of the most recent news about the world?")
