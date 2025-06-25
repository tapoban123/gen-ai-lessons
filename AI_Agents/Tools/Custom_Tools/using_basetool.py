from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="First number to multiply")
    b: int = Field(required=True, description="Second number to multiply")


class MultiplyTool(BaseTool):
    name: str = "Multiply"
    description: str = "Tool to multiply two numbers"

    args_schema: Type[MultiplyInput] = MultiplyInput

    def _run(self, a: int, b: int):
        return a * b


multiply_tool = MultiplyTool()

result = multiply_tool.invoke({"a": 5, "b": 7})

print(result)