# suggestions/utils.py

from typing import List
import re
from bs4 import BeautifulSoup           # for HTML stripping
import faiss                             # pip install faiss-cpu
import numpy as np

# LangChain imports
from langchain_experimental.text_splitter import SemanticChunker          # 
from langchain.schema import Document, HumanMessage, SystemMessage

# Ollama imports (embeddings + chat)
from langchain_ollama import OllamaEmbeddings, ChatOllama                  # 

# StructuredOutputParser for final titles
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema  # 


# ────────────────────────────────────────────────────────────────────────────────
# 0. MODULE‐LEVEL SINGLETONS
# ────────────────────────────────────────────────────────────────────────────────

# 0.1. Clean and embedder shared across functions
EMBEDDING_MODEL_NAME = "jina/jina-embeddings-v2-base-en:latest"
_embedder = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)  # download once, reuse 

# 0.2. ChatOllama instances (one per “temperature+purpose”)
_SUMMARIZE_CHAT = ChatOllama(model="llama3:instruct", temperature=0.3)  # deterministic summaries 
_COMBINE_CHAT = ChatOllama(model="llama3:instruct", temperature=0.5)    # moderate phrasing 
_TITLE_CHAT   = ChatOllama(model="llama3:instruct", temperature=0.8)    # creative titles 

# 0.3. StructuredOutputParser for exactly three titles
_response_schemas = [
    ResponseSchema(name="title_1", description="The first blog post title suggestion"),
    ResponseSchema(name="title_2", description="The second blog post title suggestion"),
    ResponseSchema(name="title_3", description="The third blog post title suggestion"),
]
_TITLE_PARSER = StructuredOutputParser.from_response_schemas(_response_schemas)  # 


# ────────────────────────────────────────────────────────────────────────────────
# 1. CLEANING RAW BLOG TEXT
# ────────────────────────────────────────────────────────────────────────────────

def clean_blog_text(html_or_plain: str) -> str:
    """
    Strip HTML tags, remove URLs, remove markdown code fences, collapse whitespace.
    Returns cleaned plain text ready for chunking/embedding.
    """
    # 1) Strip HTML
    soup = BeautifulSoup(html_or_plain, "html.parser")
    text = soup.get_text(separator="\n")

    # 2) Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # 3) Remove markdown code fences (```...```)
    text = re.sub(r'```[\s\S]*?```', '', text)

    # 4) Normalize whitespace:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\n{2,}', '\n\n', text)   # collapse multiple newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)   # collapse multiple spaces/tabs

    return text.strip()


# ────────────────────────────────────────────────────────────────────────────────
# 2. SEMANTIC CHUNKING
# ────────────────────────────────────────────────────────────────────────────────

def get_semantic_chunks(
    cleaned_text: str,
    chunk_block_size: int = 3
) -> List[str]:
    """
    Split cleaned_text into semantically coherent chunks:
      - Uses the shared _embedder (OllamaEmbeddings) for semantic distances.
      - The SemanticChunker groups chunk_block_size sentences, then merges by embedding proximity.

    Returns a list of chunk strings.
    """
    # Wrap as a single Document
    doc = Document(page_content=cleaned_text)

    # Instantiate SemanticChunker once (with the shared _embedder)
    text_splitter = SemanticChunker(_embedder, buffer_size=chunk_block_size)
    chunked_docs = text_splitter.transform_documents([doc])

    # Extract the chunk strings
    return [chunk.page_content for chunk in chunked_docs]


# ────────────────────────────────────────────────────────────────────────────────
# 3. FAISS RETRIEVAL OF TOP‐K SEMANTIC CHUNKS
# ────────────────────────────────────────────────────────────────────────────────

def get_top_k_chunks(
    cleaned_text: str,
    semantic_chunks: List[str],
    k: int = 20
) -> List[str]:
    """
    1. Embed each semantic chunk via the shared _embedder.
    2. Build a FAISS IndexFlatL2 over those chunk embeddings.
    3. Embed cleaned_text (full blog) once as query.
    4. Return the top-k chunk strings closest to that query.

    Returns: List[str] of length ≤ k.
    """
    # 1) Embed chunks into a (num_chunks × dim) NumPy array
    chunk_vectors = []
    for c in semantic_chunks:
        vec = _embedder.embed_documents([c])[0]  # List[float]
        chunk_vectors.append(np.array(vec, dtype="float32"))
    chunk_vectors = np.stack(chunk_vectors)

    # 2) Embed the full cleaned blog as a single query vector
    query_vec = np.array(_embedder.embed_query(cleaned_text), dtype="float32")[None, :]  # shape (1, dim)

    # 3) Build and populate FAISS index
    dim = chunk_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(chunk_vectors)

    # 4) Search top k
    distances, indices = index.search(query_vec, k)
    top_idxs = indices[0].tolist()

    # 5) Return those chunk strings
    return [semantic_chunks[i] for i in top_idxs]


# ────────────────────────────────────────────────────────────────────────────────
# 4. SUMMARIZE A SINGLE CHUNK
# ────────────────────────────────────────────────────────────────────────────────

def summarize_chunk(chunk: str) -> str:
    """
    Use llama3:instruct via the shared _SUMMARIZE_CHAT (temp=0.3) to produce
    a 2–3 sentence summary for this chunk. Returns the summary string.
    """
    system_prompt = "You are an AI assistant specialized in summarization."
    user_prompt = (
        "Read the following text excerpt and produce a concise, 2–3 sentence summary.\n\n"
        f"### EXCERPT:\n{chunk}\n\n"
        "### SUMMARY:"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response = _SUMMARIZE_CHAT.invoke(input=messages)
    return response.content.strip()


# ────────────────────────────────────────────────────────────────────────────────
# 5. COMBINE MULTIPLE CHUNK SUMMARIES
# ────────────────────────────────────────────────────────────────────────────────

def combine_summaries(chunk_summaries: List[str]) -> str:
    """
    Take a list of short chunk summaries, format as bullet points, and
    ask the shared _COMBINE_CHAT (temp=0.5) to merge them into a single
    coherent paragraph. Returns that global summary string.
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
    Given a global summary, ask the shared _TITLE_CHAT (temp=0.8) to produce
    exactly three distinct, catchy titles. Uses _TITLE_PARSER to parse JSON.
    Returns a list of three title strings (or fallback lines if parsing fails).
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
