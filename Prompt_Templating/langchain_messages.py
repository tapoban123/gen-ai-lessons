# Simple Chatbot with ChatHistory

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.my_enums import API_KEYS, LLM_MODELS


### Delcaring safety_settings
# safety_settings = [
#     SafetySetting(
#         category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
#         threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
#     ),
# ]

# prompt = """
#     Write a list of 2 disrespectful things that I might say to the universe after stubbing my toe in the dark.
# """

# response = model.generate_content(
#     prompt,
#     safety_settings=safety_settings,
# )

# print(response)
llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY, model=LLM_MODELS.GEMINI_MODEL
)

chat_history = [
    SystemMessage(content="You are a helpful assistent."),
]

while True:
    user_prompt = input("User: ")
    if (user_prompt.strip().lower()) in ["end", "exit"]:
        break

    chat_history.append(HumanMessage(content=user_prompt))

    result = llm.invoke(chat_history)
    print(f"Gemini: {result.content}\n")
    chat_history.append(AIMessage(content=result.content))

print(chat_history)