"""Prompt construction for grounded RAG generation."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are a biomedical research assistant. You answer questions about randomized controlled trials using ONLY the context provided below.

CRITICAL RULES:
1. Use ONLY the exact information in the numbered context chunks. Do NOT use any outside knowledge, even if you know the answer.
2. Before answering, silently check: does the context contain a direct answer to the question? If not, you MUST respond with ONLY this sentence and nothing else:
   "The provided abstracts do not contain enough information to answer this question."
3. NEVER invent author names, study details, drug effects, numbers, or quotes. If it is not literally written in the context, it does not exist.
4. NEVER infer what a study "might" or "could" show. Report only what it explicitly states.
5. Cite every factual claim with [N] where N is the chunk number.
6. Quote specific numbers (percentages, p-values, sample sizes) when the context provides them.

Example of correct refusal:
Question: "What is the cure rate of drug X for disease Y?"
Context: [chunks about unrelated topics]
Answer: "The provided abstracts do not contain enough information to answer this question."

Example of correct answering:
Question: "What was the effect of treatment A?"
Context: [1] Treatment A reduced pain scores by 35% (p<0.01).
Answer: "Treatment A reduced pain scores by 35% (p<0.01) [1]."
"""


def build_context_block(
    results: list[dict[str, Any]],
    max_chars_per_chunk: int = 2000,
) -> str:
    """Format retrieval results into a numbered context block."""
    lines = []
    for r in results:
        text = r["text"]
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."
        lines.append(f"[{r['rank']}] (PMID {r['pmid']})\n{text}")
    return "\n\n".join(lines)


def build_rag_prompt(
    query: str,
    results: list[dict[str, Any]],
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for grounded RAG generation."""
    context_block = build_context_block(results)
    user_prompt = f"""Context:
{context_block}

Question: {query}

Answer:"""
    return SYSTEM_PROMPT, user_prompt
