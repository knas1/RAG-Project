"""Settings for the standards search pipeline."""
from pathlib import Path

# Paths
PDF_DIR = Path("./pdfs")
DB_DIR = Path("./chroma_db")
COLLECTION_NAME = "standards"

# Chunking (measured in characters; ~4 chars per token)
MAX_CHUNK_CHARS = 3200
CHUNK_OVERLAP_CHARS = 400
MIN_CHUNK_CHARS = 120

# Embedding model (runs locally, no API key needed)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Hybrid retrieval
DENSE_K = 20
SPARSE_K = 20
RRF_K = 60
FUSED_K = 20

# Reranking
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_TOP_N = 6
RERANK_MIN_SCORE = 0.30

# Answer generation (used only by chatbot.py)
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
USE_LLM_GRADER = False

# Standard name and version per PDF. Key = the PDF filename.
# Files not listed here use the filename instead.
MANIFEST = {
    # "ISO-9001-2015.pdf": {"standard": "ISO 9001", "version": "2015"},
}
