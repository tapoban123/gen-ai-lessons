from langchain.prompts import PromptTemplate

template = """Act as an academic research expert. Read and digest the content of the research paper titled {title}. Produce a concise and clear summary that encapsulates the main findings, methodology, results, and implications of the study. Ensure that the summary is written in a manner that is accessible to a general audience while retaining the core insights and nuances of the original paper. Include key terms and concepts, and provide any necessary context or background information. The summary should serve as a standalone piece that gives readers a comprehensive understanding of the paper's significance without needing to read the entire document."""

# Setting validate_template to True raise error when the expected input variables are
# not provided.
prompt = PromptTemplate(
    template=template,
    input_variables=["title"],
    validate_template=True,
    name="Prompt for Research Paper Summarization",
)

prompt.save("Prompt_Templating/research_prompt.json")
