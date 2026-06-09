# PubMed RAG
[![CI](https://github.com/pateljai032/pubmed-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/pateljai032/pubmed-rag/actions/workflows/ci.yml)

A production-grade retrieval-augmented generation system over **15,000 randomized controlled trial abstracts** from the PubMed 200k RCT dataset. Uses domain-tuned biomedical embeddings, FAISS retrieval, and a grounded generation prompt with explicit refusal behavior.

Returns answers with inline citations to source PMIDs. Designed to refuse rather than hallucinate when the corpus doesn't contain enough information.

## Demo

**Query:** *Does prednisolone reduce inflammation in osteoarthritis?*

**Response:**
```json
{
  "query": "Does prednisolone reduce inflammation in osteoarthritis?",
  "answer": "Yes, prednisolone reduces inflammation in osteoarthritis. There was a clinically relevant reduction in the serum levels of IL-1, IL-6, TNF-, and hsCRP at 6 weeks in the intervention group when compared to the placebo group [2]. These differences remained significant at 12 weeks [2].",
  "sources": [
    { "rank": 1, "score": 0.926, "pmid": "25822572", "text": "..." },
    { "rank": 2, "score": 0.926, "pmid": "24293578", "text": "..." }
  ]
}
```

**Canary query** (the corpus doesn't address this directly): *Adverse effects of statins in elderly patients*

**Response:** `"The provided abstracts do not contain enough information to answer this question."`

The refusal behavior is intentional and tested — it's the difference between a trustworthy RAG system and one that confidently hallucinates.

## Architecture

```mermaid
flowchart LR
    Query[User Query] --> API[FastAPI Service]
    API --> Embedder[PubMedBERT-MS-MARCO<br/>Query Embedder]
    Embedder --> FAISS[FAISS IndexFlatIP<br/>15k vectors, 768-dim]
    FAISS --> Retriever[Top-k Chunks]
    Retriever --> Prompt[Prompt Assembly<br/>w/ refusal scaffolding]
    Prompt --> LLM[Llama 3.3 70B<br/>via Groq API]
    LLM --> Answer[Grounded Answer<br/>w/ inline citations]
```

Three swappable layers:
- **Core library** (`src/pubmed_rag/`) — ingest, embed, index, retrieve, prompt, generate
- **CLI scripts** (`scripts/`) — batch pipeline reproducibility
- **API service** (`src/pubmed_rag/api.py`) — HTTP endpoints for live use

Both a local Phi-3 generator and an API-backed generator are included; the backend is selected via config.

## Quick start (Docker)

```bash
git clone https://github.com/pateljai032/pubmed-rag.git
cd pubmed-rag

# Set up environment
cp .env.example .env
# Edit .env and add your Groq API key (free at console.groq.com)

# Run the pipeline once locally to build artifacts (one-time, ~10 min)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/setup_data.sh  # downloads PubMed 200k RCT
python scripts/01_ingest.py
python scripts/02_embed_and_index.py --split train

# Then run the service in Docker
docker compose up
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + readiness probe |
| POST | `/retrieve` | Top-k chunks for a query (no LLM call) |
| POST | `/ask` | Full RAG: retrieve + grounded answer |

Example:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"Does prednisolone reduce inflammation in osteoarthritis?","k":5}'
```

## Project structure

```
pubmed-rag/
├── src/pubmed_rag/
│   ├── ingest.py       # Parse PubMed RCT .txt → JSONL
│   ├── embed.py        # PubMedBERT embedding wrapper
│   ├── index.py        # FAISS index build/load
│   ├── retrieve.py     # Retriever class (FAISS + metadata)
│   ├── prompt.py       # Grounded RAG prompt with refusal rules
│   ├── generate.py     # Local (HuggingFace) + API generators
│   ├── config.py       # Pydantic-validated YAML config
│   └── api.py          # FastAPI app with lifespan model loading
├── scripts/            # CLI pipeline (01_ingest → 02_embed → 03_query → 04_ask)
├── tests/              # Unit tests (ingest, prompt, API)
├── configs/default.yaml
├── Dockerfile
└── docker-compose.yml
```

## Design decisions

**Domain-tuned embeddings (`pritamdeka/S-PubMedBert-MS-MARCO`)** — chosen over generic MiniLM because the corpus is biomedical and the model is fine-tuned for query→passage retrieval (MS-MARCO objective).

**FAISS `IndexFlatIP` over normalized vectors** — exact search at 15k scale is fast (~5ms per query). Normalized inner product equals cosine similarity, which is the natural semantic metric. Approximate indexes (IVF, HNSW) would be premature optimization at this scale.

**One abstract = one chunk** — abstracts are short (~1800 chars median) and structurally coherent. Per-sentence chunking is implemented in metadata for future label-filtered retrieval.

**Refusal scaffolding in the system prompt** — explicit examples of correct refusal and correct answering. Tested with a canary query that has no good answer in the corpus; the system refuses cleanly instead of hallucinating.

**Swappable generation backend** — `Generator` (local HuggingFace) and `APIGenerator` (OpenAI-compatible API: Groq, OpenAI, Together) share the same `.generate(system, user) -> str` interface. Selected via `generation.backend` in config.

**Model loaded once at startup** — FastAPI `lifespan` handler initializes embedder, retriever, and generator on server start. Per-request overhead is just retrieval (~5ms) + generation (~2s for API, ~30s for local CPU).

## Tech stack

- Python 3.11
- `sentence-transformers` (PubMedBERT)
- `faiss-cpu`
- `transformers` (local LLM backend)
- `openai` SDK (Groq-compatible API)
- `FastAPI` + `uvicorn`
- `pydantic` v2
- `pytest`, `ruff`

## Dataset

[PubMed 200k RCT](https://github.com/Franck-Dernoncourt/pubmed-rct) — sentence-level classified abstracts from PubMed. Currently using the 20k subset (15k train / 2.5k dev / 2.5k test). Each abstract carries per-sentence labels (BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS), preserved in metadata for future label-filtered retrieval.

## Limitations & future work

- **No reranker** — adding a cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) would noticeably improve top-k precision, especially for queries with qualifiers like "in elderly patients."
- **No formal evaluation set** — qualitative testing only. A labeled Q&A set with PMID ground truth would enable Recall@k and faithfulness metrics.
- **English-only**, RCT-only corpus — generalizing to the full PubMed 200k or other biomedical text would require larger indexes (IVF or HNSW).
- **No streaming** — `/ask` returns the full response after generation. Streaming via Server-Sent Events would improve perceived latency.
- **Adding Guardrails** 

## License

MIT — see [LICENSE](LICENSE).