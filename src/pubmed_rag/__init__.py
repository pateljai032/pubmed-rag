"""PubMed RAG: domain-specific retrieval-augmented generation over RCT abstracts."""
import os

# Tame OpenMP / threading conflicts between FAISS and PyTorch on macOS.
# These MUST be set before any faiss / torch imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Load faiss before torch.
import faiss  # noqa: E402, F401

__version__ = "0.1.0"