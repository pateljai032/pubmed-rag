"""CLI: run a single query against the retriever and print the assembled prompt."""

from __future__ import annotations

import argparse

from pubmed_rag.config import Config
from pubmed_rag.embed import Embedder
from pubmed_rag.prompt import build_rag_prompt
from pubmed_rag.retrieve import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--query", required=True, help="Natural-language question")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    cfg = Config.load(args.config)

    embedder = Embedder(
        model_id=cfg.embedding.model_id,
        normalize=cfg.embedding.normalize,
    )
    retriever = Retriever.from_artifacts(
        artifacts_dir=cfg.paths.artifacts_dir / args.split,
        abstracts_path=cfg.paths.data_dir / f"{args.split}_abstracts.jsonl",
        embedder=embedder,
    )

    results = retriever.retrieve(args.query, k=args.k)

    print(f"\n=== Top-{args.k} for: {args.query} ===\n")
    for r in results:
        print(f"[{r['score']:.4f}] PMID {r['pmid']}")
        print(f"  {r['text'][:180]}...\n")

    system, user = build_rag_prompt(args.query, results)
    print("=" * 70)
    print("ASSEMBLED PROMPT (system + user):")
    print("=" * 70)
    print(system)
    print("-" * 70)
    print(user)


if __name__ == "__main__":
    main()
