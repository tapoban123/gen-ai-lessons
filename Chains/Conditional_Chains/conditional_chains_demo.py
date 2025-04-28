from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.my_enums import API_KEYS, LLM_MODELS
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal


llm = ChatGoogleGenerativeAI(
    api_key=API_KEYS.GEMINI_API_KEY,
    model=LLM_MODELS.GEMINI_MODEL,
    temperature=0,
)

str_parser = StrOutputParser()


class SentimentParser(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Return the sentiment of the feedback."
    )
    feedback: str = Field(description="The actual feedback message provided by the user.")


pydantic_parser = PydanticOutputParser(pydantic_object=SentimentParser)

classifier_template = PromptTemplate(
    template="Perform sentiment analysis of the {feedback}.\n{format_instructions}",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": pydantic_parser.get_format_instructions()
    },
)

positive_template = PromptTemplate(
    template="Write one appropriate message for the positive feedback of the user.\n{feedback}",
    input_variables=["feedback"],
)

negative_template = PromptTemplate(
    template="Write one appropriate message for the negative feedback of the user.\n{feedback}",
    input_variables=["feedback"],
)

classifier_chain = classifier_template | llm | pydantic_parser

branch_chain = RunnableBranch(
    # when positive
    (lambda x: x.sentiment == "positive", positive_template | llm | str_parser),
    # when negative
    (lambda x: x.sentiment == "negative", negative_template | llm | str_parser),
    # default chain: when neither positive nor negative
    RunnableLambda(lambda x: x.sentiment == "cannot analyse sentiment from feedback"),
)

final_chain = classifier_chain | branch_chain

pos_feedback = "Quality of the shirt is really good and price is also very low."
neg_feedback = "The charger stopped working after using the mobile phone for 3 days."

result = final_chain.invoke(
    {"feedback": neg_feedback}
)

print(result)

final_chain.get_graph().print_ascii()