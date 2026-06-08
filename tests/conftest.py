"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

# A tiny sample of the PubMed RCT format, enough to exercise the parser.
SAMPLE_RCT = """###12345678
BACKGROUND\tDiabetes is a major health issue .
OBJECTIVE\tTo test a new drug .
METHODS\tWe ran a trial with 100 patients .
RESULTS\tThe drug reduced HbA1c by 1.2 % ( p < 0.001 ) .
CONCLUSIONS\tThe drug is effective .

###87654321
BACKGROUND\tHypertension affects many .
METHODS\tA crossover trial was performed .
RESULTS\tBlood pressure dropped 10 mmHg .
"""


@pytest.fixture
def sample_rct_file(tmp_path):
    """Write the sample to a temp file and return the path."""
    path = tmp_path / "sample.txt"
    path.write_text(SAMPLE_RCT)
    return path


@pytest.fixture
def sample_retrieval_results():
    """A minimal set of retrieval results for prompt-building tests."""
    return [
        {
            "rank": 1,
            "score": 0.92,
            "pmid": "12345678",
            "text": "Diabetes is a major health issue . The drug reduced HbA1c by 1.2 % .",
            "sentences": [],
        },
        {
            "rank": 2,
            "score": 0.85,
            "pmid": "87654321",
            "text": "Hypertension affects many . Blood pressure dropped 10 mmHg .",
            "sentences": [],
        },
    ]
