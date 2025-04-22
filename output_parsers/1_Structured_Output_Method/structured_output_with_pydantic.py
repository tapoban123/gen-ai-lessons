from typing import Literal, Optional
from utils.my_enums import API_KEYS, LLM_MODELS
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

user_review = "I had the most incredible dining experience at this restaurant! The food was absolutely delicious - every dish was perfectly prepared and beautifully presented. Our server was attentive, knowledgeable, and made excellent recommendations. The atmosphere was elegant yet comfortable. I can't wait to come back and try more items from their menu. Definitely a new favorite spot!"

llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY, model=LLM_MODELS.GEMINI_MODEL
)


# Schema
class Review(BaseModel):
    # With Annotated we add a helper message for the LLM to understand the task.
    summary: str = Field(description="A brief summary of the review.")
    sentiment: Literal["pos", "neg"]
    pros: Optional[list[str]] = Field(description="Write down all the pros in a list.")


structured_model = llm.with_structured_output(Review)

result = structured_model.invoke(user_review)

response = result.model_dump()

print(response, end="\n" * 2)
print(response["summary"], end="\n" * 2)
print(response["sentiment"])
