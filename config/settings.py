"""Single source of truth for every constant in the project.

Every module imports its numbers and names from here. No other file may
hard-code a path, model name, size, or threshold.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DOCS_DIR: Path = PROJECT_ROOT / "docs"
CHROMA_PATH: Path = PROJECT_ROOT / "chroma_data"
EVAL_QUESTIONS_PATH: Path = PROJECT_ROOT / "eval" / "questions.json"
DOCS_GLOB: str = "*.md"

# --- Vector store --------------------------------------------------------
COLLECTION_NAME: str = "documentation"
# Distance used by the HNSW index; cosine pairs with normalised embeddings.
DISTANCE_METRIC: str = "cosine"
# Characters of the content hash kept in each chunk ID.
CHUNK_ID_HASH_LENGTH: int = 12

# --- Chunking ------------------------------------------------------------
# Markdown heading levels: `##` opens a section, `###` opens a subsection.
SECTION_HEADING_LEVEL: int = 2
SUBSECTION_HEADING_LEVEL: int = 3
# Joins doc title, section, and subsection in the context header.
HEADING_PATH_SEPARATOR: str = " > "
# An H2 section longer than this many words is re-split on its H3 headings.
H3_SPLIT_THRESHOLD: int = 400
# Chunks smaller than this many tokens are merged into a neighbour from the
# same `##` section; too little text to rank well on its own.
MIN_CHUNK_TOKENS: int = 40
# Joins subsection labels when small chunks are merged.
SUBSECTION_JOINER: str = " / "

# --- Embedding -----------------------------------------------------------
# bge-small reads up to 512 tokens (all-MiniLM-L6-v2 stops at 256).
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
# bge-small's input limit. The chunker subtracts each chunk's heading-path
# length from this, so prefix + chunk always fits without truncation.
MAX_EMBED_TOKENS: int = 512
# bge expects this instruction in front of search queries (not documents).
EMBED_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

# --- Retrieval -----------------------------------------------------------
TOP_K: int = 5
# How many results each retriever (vector, BM25) contributes before fusion.
SEARCH_CANDIDATES: int = 20
# Standard Reciprocal Rank Fusion constant from the original RRF paper.
RRF_K: int = 60
# Cross-encoder that re-scores the fused candidates (reads query + chunk together).
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# How many fused candidates the reranker re-scores before keeping TOP_K.
RERANK_CANDIDATES: int = 20
# The reranker's input limit for query + heading path + chunk.
RERANKER_MAX_TOKENS: int = 512

# --- Generation ----------------------------------------------------------
# DeepSeek's API is OpenAI-compatible, so the `openai` SDK talks to it directly.
LLM_BASE_URL: str = "https://api.deepseek.com"
LLM_API_KEY_ENV: str = "DEEPSEEK_API_KEY"
LLM_MODEL: str = "deepseek-chat"
MAX_TOKENS: int = 1024
# 0 keeps answers deterministic and close to the retrieved text.
LLM_TEMPERATURE: float = 0.0
# The exact reply when the retrieved chunks do not contain the answer.
REFUSAL_MESSAGE: str = "I can't answer that from the documentation."
# Question used by the search and ask demos.
DEMO_QUESTION: str = "How do I install the project?"

# --- Chat UI -------------------------------------------------------------
UI_TITLE: str = "Documentation Assistant"
UI_DESCRIPTION: str = "Ask a question about the docs. Answers come only from the documentation, with sources."
# 127.0.0.1 keeps the page on this computer; other devices cannot open it.
UI_HOST: str = "127.0.0.1"
UI_PORT: int = 7860
# URL path of the page that shows a full document; sources link here.
DOC_ROUTE: str = "/doc"
# Joins section and subsection in a heading's link anchor, e.g. "server-setup--authentication".
ANCHOR_SEPARATOR: str = "--"
# Clickable starter questions shown under the chat box.
UI_EXAMPLES: list[str] = [
    "How do I install the SDK?",
    "How do remote servers authenticate clients?",
    "What should I do if I get HTTP 429?",
]

# --- Evaluation ----------------------------------------------------------
# Minimum fraction of eval questions whose expected section must be in the top-k.
EVAL_HIT_TARGET: float = 0.85
# Prefix for the throwaway in-memory collections the comparison builds.
EVAL_COLLECTION_PREFIX: str = "eval_"
