"""Typed configuration loaded from YAML."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class PathsConfig(BaseModel):
    data_dir: Path
    artifacts_dir: Path
    raw_train: Path
    raw_dev: Path
    raw_test: Path


class EmbeddingConfig(BaseModel):
    model_id: str
    batch_size: int
    normalize: bool


class RetrievalConfig(BaseModel):
    k: int


class GenerationConfig(BaseModel):
    backend: str
    model_id: str
    api_base_url: str = "https://api.groq.com/openai/v1"
    max_new_tokens: int
    do_sample: bool


class Config(BaseModel):
    paths: PathsConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load and validate config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
