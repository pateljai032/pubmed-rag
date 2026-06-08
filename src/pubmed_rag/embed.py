"""Embedding model wrapper around sentence-transformers."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wraps a sentence-transformers model for document and query embedding.

    Same model is used for both - critical for retrieval consistency.
    """

    def __init__(self, model_id: str, normalize: bool = True) -> None:
        self.model = SentenceTransformer(model_id)
        self.normalize = normalize
        self.dim: int = self.model.get_embedding_dimension()

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts. Returns shape (n, dim), float32."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns shape (1, dim) - ready for FAISS."""
        vec = self.embed_texts([query], show_progress=False)
        return vec.reshape(1, -1)