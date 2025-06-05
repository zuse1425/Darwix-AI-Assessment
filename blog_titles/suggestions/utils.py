import os
from typing import List
import re
from bs4 import BeautifulSoup           # pip install beautifulsoup4
import faiss                             # pip install faiss-cpu
import numpy as np

# Hugging Face summarization pipeline and tokenizer
from transformers import pipeline, AutoTokenizer   # pip install transformers[sentencepiece]
import torch

# LangChain for text splitting and schema definitions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, HumanMessage, SystemMessage

# Ollama for embeddings and chat models
from langchain_ollama import OllamaEmbeddings, ChatOllama

# StructuredOutputParser for enforcing three-title JSON output
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema


# ────────────────────────────────────────────────────────────────────────────────
# MODULE‐LEVEL SINGLETONS
# ────────────────────────────────────────────────────────────────────────────────

# 1. HF summarizer + tokenizer (use DistilBART to handle ≤1024‐token inputs safely)
_HF_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
_HF_TOKENIZER = AutoTokenizer.from_pretrained(_HF_MODEL_NAME)

if torch.cuda.is_available():
    _HF_DEVICE = 0    # use GPU 0
    os.environ["OLLAMA_DEFAULT_DEVICE"]    = "cuda"
    os.environ["OLLAMA_DEFAULT_PRECISION"] = "fp16"
else:
    _HF_DEVICE = -1   # fallback to CPU

_HF_SUMMARIZER = pipeline(
    "summarization",
    model=_HF_MODEL_NAME,
    tokenizer=_HF_MODEL_NAME,
    device=_HF_DEVICE
)
# 2. Ollama embedding model for FAISS retrieval
EMBEDDING_MODEL_NAME = "jina/jina-embeddings-v2-base-en:latest"
_embedder = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

# 3. Ollama ChatOllama instances at fixed temperatures
_COMBINE_CHAT = ChatOllama(model="llama3:instruct", temperature=0.5)
_TITLE_CHAT   = ChatOllama(model="llama3:instruct", temperature=0.8)

# 4. StructuredOutputParser enforcing exactly three titles
_response_schemas = [
    ResponseSchema(name="title_1", description="The first blog post title suggestion"),
    ResponseSchema(name="title_2", description="The second blog post title suggestion"),
    ResponseSchema(name="title_3", description="The third blog post title suggestion"),
]
_TITLE_PARSER = StructuredOutputParser.from_response_schemas(_response_schemas)


# ────────────────────────────────────────────────────────────────────────────────
# 1. CLEANING RAW BLOG TEXT
# ────────────────────────────────────────────────────────────────────────────────

def clean_blog_text(html_or_plain: str) -> str:
    """
    Strip HTML tags, remove URLs, remove markdown code fences, and collapse whitespace.
    Returns cleaned plain text ready for splitting and embedding.
    """
    # 1) Strip HTML tags and convert entities
    soup = BeautifulSoup(html_or_plain, "html.parser")
    text = soup.get_text(separator="\n")

    # 2) Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # 3) Remove markdown code fences (```...```)
    text = re.sub(r'```[\s\S]*?```', '', text)

    # 4) Normalize whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\n{2,}', '\n\n', text)   # collapse multiple newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)   # collapse multiple spaces/tabs

    return text.strip()


# ────────────────────────────────────────────────────────────────────────────────
# 2. RECURSIVE CHARACTER SPLITTING
# ────────────────────────────────────────────────────────────────────────────────

def get_recursive_chunks(
    cleaned_text: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    separators: List[str] = None
) -> List[str]:
    """
    Split cleaned_text into overlapping chunks of up to chunk_size characters,
    using RecursiveCharacterTextSplitter. Each chunk will be ≤ chunk_size characters
    (plus some negligible whitespace), and adjacent chunks share chunk_overlap characters.

    Args:
      cleaned_text:   The HTML‐stripped, whitespace‐normalized blog text.
      chunk_size:     Maximum characters per chunk (default 1024).
      chunk_overlap:  Characters of overlap between consecutive chunks (default 200).
      separators:     Ordered list of separators for splitting; if None, defaults to
                      ["\n\n", "\n", " ", ""], meaning: first try paragraph splits,
                      then line breaks, then spaces, then fallback to raw character.

    Returns:
      A list of string chunks, each ≤ chunk_size characters, overlapping by chunk_overlap.
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    return splitter.split_text(cleaned_text)


# ────────────────────────────────────────────────────────────────────────────────
# 3. FAISS RETRIEVAL OF TOP‐K CHUNKS
# ────────────────────────────────────────────────────────────────────────────────

def get_top_k_chunks(
    cleaned_text: str,
    chunks: List[str],
    k: int = 20
) -> List[str]:
    """
    1. Embed each chunk via the shared _embedder.
    2. Build a FAISS IndexFlatL2 over those chunk embeddings.
    3. Embed cleaned_text (full blog) once as a query.
    4. Return the top k chunk strings closest to that query.

    Returns: List[str] of length ≤ k.
    """
    # 1) Embed all chunks into a NumPy array of shape (N, dim)
    chunk_vectors = []
    for c in chunks:
        vec = _embedder.embed_documents([c])[0]  # returns List[float]
        chunk_vectors.append(np.array(vec, dtype="float32"))
    chunk_vectors = np.stack(chunk_vectors)

    # 2) Embed the full cleaned blog as a single query vector
    query_vec = np.array(_embedder.embed_query(cleaned_text), dtype="float32")[None, :]

    # 3) Build and populate a FAISS index (L2 distance)
    dim = chunk_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(chunk_vectors)

    # 4) Search for k nearest neighbors
    distances, indices = index.search(query_vec, k)
    top_idxs = indices[0].tolist()

    # 5) Return those chunk strings
    return [chunks[i] for i in top_idxs]


# ────────────────────────────────────────────────────────────────────────────────
# 4. SUMMARIZE A SINGLE CHUNK (using HF DistilBART)
# ────────────────────────────────────────────────────────────────────────────────

def summarize_chunk(chunk: str) -> str:
    """
    Summarize one chunk (assumed ≤ ~1024 tokens) using DistilBART on GPU 0 (if available).
    Dynamically chooses max_length so it never exceeds half the input token length.
    """
    # 1) Tokenize to see how many tokens the chunk contains
    encoding = _HF_TOKENIZER(
        chunk,
        return_tensors="pt",
        truncation=True,
        max_length=_HF_TOKENIZER.model_max_length  # 1024 for DistilBART
    )
    input_len = encoding["input_ids"].shape[-1]

    # 2) Pick max_length = min(150, input_len//2), min_length = min(30, max_length//2)
    max_length = min(150, input_len // 2)
    min_length = min(30, max_length // 2)

    # 3) Run the summarizer (on GPU 0 if available)
    summary_list = _HF_SUMMARIZER(
        chunk,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True
    )

    return summary_list[0]["summary_text"].strip()


# ────────────────────────────────────────────────────────────────────────────────
# 5. COMBINE MULTIPLE CHUNK SUMMARIES
# ────────────────────────────────────────────────────────────────────────────────

def combine_summaries(chunk_summaries: List[str]) -> str:
    """
    Merge bullet-point summaries into one coherent paragraph using llama3:instruct.
    Temperature = 0.5 for balanced phrasing.
    """
    bullets = "\n".join(f"- {s}" for s in chunk_summaries)
    system_prompt = "You are an AI assistant that synthesizes bullet-point summaries into a coherent paragraph."
    user_prompt = (
        "Below are bullet-point summaries of different sections of a blog post. "
        "Create a single paragraph that captures the entire blog’s main points.\n\n"
        f"{bullets}\n\n"
        "### GLOBAL SUMMARY:"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response = _COMBINE_CHAT.invoke(input=messages)
    return response.content.strip()


# ────────────────────────────────────────────────────────────────────────────────
# 6. GENERATE THREE DISTINCT TITLES
# ────────────────────────────────────────────────────────────────────────────────

def generate_three_titles(global_summary: str) -> List[str]:
    """
    Generate three distinct, engaging titles from the global summary using llama3:instruct.
    Temperature = 0.8 for creativity. Parsed via _TITLE_PARSER.
    """
    system_prompt = "You are an AI assistant that specializes in generating catchy blog post titles."
    format_instructions = _TITLE_PARSER.get_format_instructions()
    user_prompt = (
        "Based on the following concise summary, provide exactly three distinct, engaging titles.\n\n"
        f"### GLOBAL SUMMARY:\n{global_summary}\n\n"
        f"{format_instructions}"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    raw_response = _TITLE_CHAT.invoke(input=messages)

    try:
        parsed = _TITLE_PARSER.parse(raw_response.content.strip())
        return [parsed["title_1"], parsed["title_2"], parsed["title_3"]]
    except Exception:
        lines = [line.strip() for line in raw_response.content.splitlines() if line.strip()]
        return lines[:3]
