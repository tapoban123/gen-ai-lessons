import gradio as gr


def letter_counter(word: str, letter: str) -> int:
    """Count the number of occurrences of a letter in a word or text."""
    word = word.lower()
    letter = letter.lower()
    count = word.count(letter)
    return count


demo = gr.Interface(
    fn=letter_counter,
    inputs=["textbox", "t"],
    outputs="number",
    title="Letter Counter",
    description="Enter text and a letter to count how many times the letter appears in the text.",
)

demo.launch()
