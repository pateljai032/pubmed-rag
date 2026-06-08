"""CLI: ask a question and get a generated, cited answer."""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from pubmed_rag.config import Config
from pubmed_rag.embed import Embedder
from pubmed_rag.generate import APIGenerator, Generator
from pubmed_rag.prompt import build_rag_prompt
from pubmed_rag.retrieve import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    cfg = Config.load(args.config)

    print("Loading embedder...")
    embedder = Embedder(
        model_id=cfg.embedding.model_id,
        normalize=cfg.embedding.normalize,
    )

    print("Loading retriever...")
    retriever = Retriever.from_artifacts(
        artifacts_dir=cfg.paths.artifacts_dir / args.split,
        abstracts_path=cfg.paths.data_dir / f"{args.split}_abstracts.jsonl",
        embedder=embedder,
    )

    print(f"Loading generator: {cfg.generation.model_id} ({cfg.generation.backend})")
    if cfg.generation.backend == "api":
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        generator = APIGenerator(
            model_id=cfg.generation.model_id,
            api_key=api_key,
            base_url=cfg.generation.api_base_url,
            max_new_tokens=cfg.generation.max_new_tokens,
        )
    else:
        generator = Generator(
            model_id=cfg.generation.model_id,
            max_new_tokens=cfg.generation.max_new_tokens,
            do_sample=cfg.generation.do_sample,
        )

    print(f"\n=== Query: {args.query} ===\n")
    results = retriever.retrieve(args.query, k=args.k)
    for r in results:
        print(f"  [{r['score']:.4f}] PMID {r['pmid']}")

    print("\nGenerating answer...")
    system, user = build_rag_prompt(args.query, results)
    answer = generator.generate(system, user)

    print(f"\n=== Answer ===\n{answer}\n")


if __name__ == "__main__":
    main()
