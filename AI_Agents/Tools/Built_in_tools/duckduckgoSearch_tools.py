from langchain_community.tools import DuckDuckGoSearchRun


tool = DuckDuckGoSearchRun()

result = tool.invoke("Latest news about AI")

print(result)