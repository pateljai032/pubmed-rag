"""Tests for the ingest module."""
from __future__ import annotations

from pubmed_rag.ingest import parse_pubmed_rct


def test_parses_correct_number_of_abstracts(sample_rct_file):
    abstracts = parse_pubmed_rct(sample_rct_file)
    assert len(abstracts) == 2


def test_extracts_pmid_correctly(sample_rct_file):
    abstracts = parse_pubmed_rct(sample_rct_file)
    assert abstracts[0]["pmid"] == "12345678"
    assert abstracts[1]["pmid"] == "87654321"


def test_extracts_labels_and_sentences(sample_rct_file):
    abstracts = parse_pubmed_rct(sample_rct_file)
    first = abstracts[0]
    labels = [s["label"] for s in first["sentences"]]
    assert labels == ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]


def test_text_field_joins_all_sentences(sample_rct_file):
    abstracts = parse_pubmed_rct(sample_rct_file)
    text = abstracts[0]["text"]
    assert "Diabetes" in text
    assert "HbA1c" in text


def test_handles_incomplete_label_set(sample_rct_file):
    """Second abstract is missing OBJECTIVE and CONCLUSIONS — parser should still work."""
    abstracts = parse_pubmed_rct(sample_rct_file)
    second = abstracts[1]
    assert "OBJECTIVE" not in second["labels_present"]
    assert "BACKGROUND" in second["labels_present"]