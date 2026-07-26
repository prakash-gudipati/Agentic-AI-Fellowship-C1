"""
Session 33 — naive_rag.py

The baseline: one embed, one retrieve, one generation. No agent. No
quality gate. No re-querying.

This file exists so the walkthrough has an HONEST side-by-side
comparison against the agentic loop. Same corpus. Same vector store.
Same LLM. The only difference is the orchestration discipline.

Pedagogical role:
  - Slide deck shows naive on the left, agentic on the right.
  - Demo 2 runs both against the same poorly-phrased question and shows
    the agentic version winning.
  - This file is intentionally short. Naive RAG is what we are LEAVING.
"""

from __future__ import annotations

import time
from typing import Optional

from llm_client import LLMClient
from prompts import NAIVE_RAG_SYSTEM
from rag_types import NaiveAnswer
from retrieval_tools import search_kb


DEFAULT_K = 4


def answer(
    user_question: str,
    llm: Optional[LLMClient] = None,
    k: int = DEFAULT_K,
) -> NaiveAnswer:
    """Single-shot retrieve+generate. No agent. No re-querying."""

    llm = llm or LLMClient()
    started = time.perf_counter()

    # ONE retrieval call with the user's raw query.
    chunks = search_kb(user_question, k=k)

    # ONE generation call with the chunks stuffed into the prompt.
    if chunks:
        ctx = "\n\n---\n\n".join(
            f"[{c.source}]\n{c.text}" for c in chunks
        )
    else:
        ctx = "(no chunks retrieved)"

    text = llm.generate_final(
        system=NAIVE_RAG_SYSTEM,
        user_question=user_question,
        retrieved_text=ctx,
    )

    elapsed = time.perf_counter() - started
    return NaiveAnswer(
        user_question=user_question,
        answer_text=text,
        chunks=chunks,
        elapsed_seconds=elapsed,
    )
