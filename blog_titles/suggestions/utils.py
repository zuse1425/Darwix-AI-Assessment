# suggestions/utils.py

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema


# ────────────────────────────────────────────────────────────────────────────────
# 1. RECURSIVE CHARACTER TEXT SPLITTING
# ────────────────────────────────────────────────────────────────────────────────

def get_recursive_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] = None
) -> List[str]:
    """
    Split `text` into chunks of ~chunk_size characters (with chunk_overlap), using
    RecursiveCharacterTextSplitter.
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    return splitter.split_text(text)


# ────────────────────────────────────────────────────────────────────────────────
# 2. SUMMARIZE EACH CHUNK USING ChatOllama (llama3:instruct, temperature=0.3)
# ────────────────────────────────────────────────────────────────────────────────

def summarize_chunk(chunk: str) -> str:
    """
    Summarize one chunk (2–3 sentences) with llama3:instruct (temperature=0.3).
    """
    system_prompt = "You are an AI assistant specialized in summarization."
    user_prompt = (
        "Read the following text excerpt and produce a concise, 2–3 sentence summary.\n\n"
        f"### EXCERPT:\n{chunk}\n\n"
        "### SUMMARY:"
    )

    chat_model = ChatOllama(model="llama3:instruct", temperature=0.3)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response = chat_model.invoke(input=messages)
    return response.content.strip()


# ────────────────────────────────────────────────────────────────────────────────
# 3. COMBINE ALL CHUNK SUMMARIES INTO A SINGLE GLOBAL SUMMARY
#    (llama3:instruct, temperature=0.5)
# ────────────────────────────────────────────────────────────────────────────────

def combine_summaries(chunk_summaries: List[str]) -> str:
    """
    Merge bullet-point summaries into one paragraph with llama3:instruct (temperature=0.5).
    """
    bullets = "\n".join(f"- {s}" for s in chunk_summaries)
    system_prompt = "You are an AI assistant that synthesizes bullet-point summaries into a coherent paragraph."
    user_prompt = (
        "Below are bullet-point summaries of different sections of a blog post. "
        "Create a single paragraph that captures the entire blog’s main points.\n\n"
        f"{bullets}\n\n"
        "### GLOBAL SUMMARY:"
    )

    chat_model = ChatOllama(model="llama3:instruct", temperature=0.5)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response = chat_model.invoke(input=messages)
    return response.content.strip()


# ────────────────────────────────────────────────────────────────────────────────
# 4. DEFINE ResponseSchemas FOR THREE TITLES (unchanged)
# ────────────────────────────────────────────────────────────────────────────────

response_schemas = [
    ResponseSchema(name="title_1", description="The first blog post title suggestion"),
    ResponseSchema(name="title_2", description="The second blog post title suggestion"),
    ResponseSchema(name="title_3", description="The third blog post title suggestion"),
]

structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)


# ────────────────────────────────────────────────────────────────────────────────
# 5. GENERATE THREE DISTINCT TITLES USING StructuredOutputParser
#    (llama3:instruct, temperature=0.8)
# ────────────────────────────────────────────────────────────────────────────────

def generate_three_titles(global_summary: str) -> List[str]:
    """
    Generate three distinct, catchy titles using llama3:instruct (temperature=0.8).
    Parse via StructuredOutputParser.
    """
    system_prompt = "You are an AI assistant that specializes in generating catchy blog post titles."
    format_instructions = structured_parser.get_format_instructions()
    user_prompt = (
        "Based on the following concise summary, provide exactly three distinct, engaging titles.\n\n"
        f"### GLOBAL SUMMARY:\n{global_summary}\n\n"
        f"{format_instructions}"
    )

    chat_model = ChatOllama(model="llama3:instruct", temperature=0.8)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    raw_response = chat_model.invoke(input=messages)

    try:
        parsed = structured_parser.parse(raw_response.content.strip())
        return [parsed["title_1"], parsed["title_2"], parsed["title_3"]]
    except Exception:
        lines = [line.strip() for line in raw_response.content.splitlines() if line.strip()]
        return lines[:3]
