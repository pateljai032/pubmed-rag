"""Tests for the prompt module."""

from __future__ import annotations

from pubmed_rag.prompt import SYSTEM_PROMPT, build_context_block, build_rag_prompt


def test_context_block_includes_all_chunks(sample_retrieval_results):
    block = build_context_block(sample_retrieval_results)
    assert "[1]" in block
    assert "[2]" in block
    assert "12345678" in block
    assert "87654321" in block


def test_context_block_truncates_long_chunks():
    long_chunk = [
        {"rank": 1, "score": 1.0, "pmid": "0", "text": "x" * 5000, "sentences": []},
    ]
    block = build_context_block(long_chunk, max_chars_per_chunk=100)
    assert "..." in block
    assert len(block) < 500


def test_build_rag_prompt_returns_two_parts(sample_retrieval_results):
    system, user = build_rag_prompt("test query", sample_retrieval_results)
    assert system == SYSTEM_PROMPT
    assert "test query" in user
    assert "Context:" in user


def test_system_prompt_includes_refusal_instruction():
    assert "do not contain enough information" in SYSTEM_PROMPT.lower()
