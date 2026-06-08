"""CLI: parse raw PubMed RCT .txt files into JSONL."""
from __future__ import annotations

import argparse

from pubmed_rag.config import Config
from pubmed_rag.ingest import parse_pubmed_rct, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)

    splits = [
        ("train", cfg.paths.raw_train),
        ("dev", cfg.paths.raw_dev),
        ("test", cfg.paths.raw_test),
    ]

    for name, src in splits:
        if not src.exists():
            print(f"[skip] {name}: {src} not found")
            continue

        abstracts = parse_pubmed_rct(src)
        out = cfg.paths.data_dir / f"{name}_abstracts.jsonl"
        save_jsonl(abstracts, out)
        print(f"[ok]   {name}: {len(abstracts)} abstracts -> {out}")


if __name__ == "__main__":
    main()