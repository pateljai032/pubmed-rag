"""Parse PubMed 200k RCT text files into structured JSONL records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_pubmed_rct(filepath: Path) -> list[dict[str, Any]]:
    """Parse a PubMed RCT .txt file into a list of structured abstract dicts.

    Each abstract has the form:
        {
            "pmid": str,
            "text": str,                # all sentences joined with spaces
            "sentences": [{"label": str, "text": str}, ...],
            "labels_present": [str, ...]
        }
    """
    abstracts: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("###"):
                if current is not None:
                    abstracts.append(_finalize(current))
                current = {"pmid": line[3:].strip(), "sentences": []}
                continue

            if not line.strip():
                if current is not None:
                    abstracts.append(_finalize(current))
                    current = None
                continue

            if current is not None and "\t" in line:
                label, text = line.split("\t", 1)
                current["sentences"].append(
                    {"label": label.strip(), "text": text.strip()}
                )

    if current is not None:
        abstracts.append(_finalize(current))

    return abstracts


def _finalize(abstract: dict[str, Any]) -> dict[str, Any]:
    """Add joined text and label summary to an abstract dict."""
    abstract["text"] = " ".join(s["text"] for s in abstract["sentences"])
    abstract["labels_present"] = sorted({s["label"] for s in abstract["sentences"]})
    return abstract


def save_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts to a JSONL file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]