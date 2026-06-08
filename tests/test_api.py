"""Tests for the API module — route registration and schema validation."""

from __future__ import annotations

from pubmed_rag.api import app


def test_app_has_required_routes():
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/retrieve" in paths
    assert "/ask" in paths


def test_app_metadata():
    assert app.title == "PubMed RAG"
    assert app.version == "0.1.0"
