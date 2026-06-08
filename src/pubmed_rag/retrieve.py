"""Retriever: combines FAISS index + embedder + metadata for query-driven search."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss

from pubmed_rag.embed import Embedder
from pubmed_rag.index import load_index
from pubmed_rag.ingest import load_jsonl


class Retriever:
    """Combines a FAISS index, an embedder, and document metadata.

    Alignment invariant: abstracts[i] corresponds to FAISS vector id i.
    """

    def __init__(
        self,
        index: faiss.Index,
        embedder: Embedder,
        abstracts: list[dict[str, Any]],
    ) -> None:
        if index.ntotal != len(abstracts):
            raise ValueError(
                f"Misalignment: index has {index.ntotal} vectors "
                f"but {len(abstracts)} abstracts loaded"
            )
        self.index = index
        self.embedder = embedder
        self.abstracts = abstracts

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Path,
        abstracts_path: Path,
        embedder: Embedder,
    ) -> "Retriever":
        """Build a Retriever from on-disk artifacts."""
        index = load_index(artifacts_dir / "abstracts.faiss")
        abstracts = load_jsonl(abstracts_path)
        return cls(index=index, embedder=embedder, abstracts=abstracts)

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k abstracts most similar to query."""
        q_vec = self.embedder.embed_query(query)
        scores, ids = self.index.search(q_vec, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0], strict=True)):
            a = self.abstracts[int(idx)]
            results.append(
                {
                    "rank": rank + 1,
                    "score": float(score),
                    "pmid": a["pmid"],
                    "text": a["text"],
                    "sentences": a["sentences"],
                }
            )
        return results