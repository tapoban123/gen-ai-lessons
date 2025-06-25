from langchain.tools import tool

@tool
def add(a: int, b: int):
    """Method to get the sum of two numbers."""
    return a + b


result = add.invoke({"a": 5, "b": 10})

print(result)

print("\nTool Details:")
print(add.name)
print(add.description)
print(add.args)