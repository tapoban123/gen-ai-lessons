from langchain.text_splitter import CharacterTextSplitter

text = """The field of quantum computing contains a range of disciplines, including quantum hardware and quantum algorithms. While still in development, quantum technology will soon be able to solve complex problems that supercomputers can't solve, or can't solve fast enough.

By taking advantage of quantum physics, fully realized quantum computers would be able to process massively complicated problems at orders of magnitude faster than modern machines. For a quantum computer, challenges that might take a classical computer thousands of years to complete might be reduced to a matter of minutes.

The study of subatomic particles, also known as quantum mechanics, reveals unique and fundamental natural principles. Quantum computers harness these fundamental phenomena to compute probabilistically and quantum mechanically."""


splitter = CharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
    separator=""
)

result = splitter.split_text(text=text)

print(result)