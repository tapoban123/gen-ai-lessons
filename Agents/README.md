# Agents

## Here we will use serp api. By using [Serp API](https://serpapi.com/) we will call google-search-engine and extract real-time information.

## Notes

1. Setting the `verbose` parameter to `True` will show us the detail of all the processing that happens in the background.
   _E.g._:
   ```python
   self_ask_with_search = initialize_agent(
       tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
   )
   ```
