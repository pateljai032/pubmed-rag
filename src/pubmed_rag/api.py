"""FastAPI service exposing the RAG pipeline as HTTP endpoints."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pubmed_rag.config import Config
from pubmed_rag.embed import Embedder
from pubmed_rag.generate import APIGenerator, Generator
from pubmed_rag.prompt import build_rag_prompt
from pubmed_rag.retrieve import Retriever

# Shared state populated once at startup.
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and indices once on startup; clean up on shutdown."""
    load_dotenv()
    cfg = Config.load(os.environ.get("CONFIG_PATH", "configs/default.yaml"))

    print("Loading embedder...")
    embedder = Embedder(
        model_id=cfg.embedding.model_id,
        normalize=cfg.embedding.normalize,
    )

    print("Loading retriever...")
    retriever = Retriever.from_artifacts(
        artifacts_dir=cfg.paths.artifacts_dir / "train",
        abstracts_path=cfg.paths.data_dir / "train_abstracts.jsonl",
        embedder=embedder,
    )

    print(f"Loading generator: {cfg.generation.model_id} ({cfg.generation.backend})")
    if cfg.generation.backend == "api":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment")
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

    state["cfg"] = cfg
    state["retriever"] = retriever
    state["generator"] = generator
    print("Ready to serve requests.")

    yield

    state.clear()
    print("Shutdown complete.")


app = FastAPI(
    title="PubMed RAG",
    description="Retrieval-augmented generation over PubMed 200k RCT abstracts.",
    version="0.1.0",
    lifespan=lifespan,
)


# Request / response schemas
class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Natural-language question")
    k: int = Field(5, ge=1, le=20, description="Number of results to return")


class RetrievedChunk(BaseModel):
    rank: int
    score: float
    pmid: str
    text: str


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]


class AskRequest(BaseModel):
    query: str = Field(..., description="Natural-language question")
    k: int = Field(5, ge=1, le=20)


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[RetrievedChunk]


@app.get("/health")
async def health():
    """Liveness + readiness probe."""
    return {"status": "ok", "ready": "retriever" in state}


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(req: RetrieveRequest):
    """Return top-k retrieved chunks for a query (no LLM call)."""
    retriever = state.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    results = retriever.retrieve(req.query, k=req.k)
    chunks = [
        RetrievedChunk(rank=r["rank"], score=r["score"], pmid=r["pmid"], text=r["text"])
        for r in results
    ]
    return RetrieveResponse(query=req.query, results=chunks)


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    """Full RAG: retrieve + generate a grounded, cited answer."""
    retriever = state.get("retriever")
    generator = state.get("generator")
    if retriever is None or generator is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    results = retriever.retrieve(req.query, k=req.k)
    system, user = build_rag_prompt(req.query, results)
    answer = generator.generate(system, user)

    chunks = [
        RetrievedChunk(rank=r["rank"], score=r["score"], pmid=r["pmid"], text=r["text"])
        for r in results
    ]
    return AskResponse(query=req.query, answer=answer, sources=chunks)