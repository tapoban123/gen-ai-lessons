from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: int = Field(
        description="First number to multiply",
        required=True,
    )
    b: int = Field(
        description="First number to multiply",
        required=True,
    )


def multiply(a: int, b: int) -> int:
    """Method to calculate the product of two numbers."""
    return a * b


multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="Multiply",
    description="Multiply two numbers.",
    args_schema=MultiplyInput,
)

result = multiply_tool.invoke({"a": 5, "b": 5})
print(result)
