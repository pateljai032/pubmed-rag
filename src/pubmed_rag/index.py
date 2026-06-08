"""FAISS index helpers."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a flat inner-product index from normalized embeddings.

    Since vectors are normalized, inner product == cosine similarity.
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    """Persist a FAISS index to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    """Load a FAISS index from disk."""
    return faiss.read_index(str(path))