"""CLI: embed abstracts and build the FAISS index."""
from __future__ import annotations

import argparse

import numpy as np

from pubmed_rag.config import Config
from pubmed_rag.embed import Embedder
from pubmed_rag.index import build_index, save_index
from pubmed_rag.ingest import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train", help="train | dev | test")
    args = parser.parse_args()

    cfg = Config.load(args.config)

    # 1. Load abstracts
    abstracts_path = cfg.paths.data_dir / f"{args.split}_abstracts.jsonl"
    abstracts = load_jsonl(abstracts_path)
    texts = [a["text"] for a in abstracts]
    print(f"Loaded {len(texts)} abstracts from {abstracts_path}")

    # 2. Embed
    embedder = Embedder(
        model_id=cfg.embedding.model_id,
        normalize=cfg.embedding.normalize,
    )
    print(f"Embedder: {cfg.embedding.model_id} (dim={embedder.dim})")

    embeddings = embedder.embed_texts(texts, batch_size=cfg.embedding.batch_size)
    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # 3. Save embeddings + PMIDs (alignment between FAISS id i and PMID is critical)
    out_dir = cfg.paths.artifacts_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings -> {emb_path}")

    pmid_path = out_dir / "pmids.txt"
    pmid_path.write_text("\n".join(a["pmid"] for a in abstracts))
    print(f"Saved PMIDs -> {pmid_path}")

    # 4. Build + save index
    index = build_index(embeddings)
    index_path = out_dir / "abstracts.faiss"
    save_index(index, index_path)
    print(f"Saved index -> {index_path} ({index.ntotal} vectors)")


if __name__ == "__main__":
    main()