"""
Session 33 — rag_types.py

Typed records shared across the agentic RAG pipeline.

Why typed records (and not raw dicts)?
  - The retrieval loop crosses several modules (tools, planner, quality
    gate, agent). Passing typed objects makes it obvious WHAT each module
    is allowed to read and write.
  - The walkthrough format means students read the modules in any order.
    Types are the cheapest documentation we get for free.

Nothing in this file talks to ChromaDB, the LLM, or the network. Pure data
and helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ----------------------------------------------------------------------------
# RetrievedChunk — what comes back from a single search_kb call
# ----------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """One chunk pulled out of the vector store.

    Fields:
      chunk_id   — stable identifier ("doc:offset")
      text       — the raw chunk text
      source     — filename the chunk came from (corpus/02_pricing.md)
      distance   — Chroma distance score (lower = closer in vector space)
      relevance  — 1..5 score assigned by the quality gate. None until
                   the gate has scored this chunk.
    """

    chunk_id: str
    text: str
    source: str
    distance: float
    relevance: Optional[int] = None

    def short(self) -> str:
        """One-line preview used by the trace logger."""

        head = self.text.replace("\n", " ").strip()
        if len(head) > 90:
            head = head[:87] + "..."
        rel = "?" if self.relevance is None else str(self.relevance)
        return f"[{rel}/5] {self.source} :: {head}"


# ----------------------------------------------------------------------------
# RetrievalAttempt — one pass through search_kb with a single query
# ----------------------------------------------------------------------------


@dataclass
class RetrievalAttempt:
    """A single search_kb invocation plus its quality verdict.

    The agentic loop produces one of these per retrieval call. The
    sequence of attempts is the trace the instructor walks through on
    screen.
    """

    attempt_index: int
    query: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    average_relevance: Optional[float] = None
    gate_passed: bool = False
    notes: str = ""


# ----------------------------------------------------------------------------
# AgenticAnswer — final output from the agentic RAG run
# ----------------------------------------------------------------------------


@dataclass
class AgenticAnswer:
    """End-of-turn record. Everything the walkthrough needs in one object."""

    user_question: str
    answer_text: str
    attempts: List[RetrievalAttempt] = field(default_factory=list)
    retrieval_budget: int = 0
    budget_exhausted: bool = False
    decided_no_retrieval: bool = False
    decomposed_sub_questions: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def total_chunks_seen(self) -> int:
        return sum(len(a.chunks) for a in self.attempts)

    def total_attempts(self) -> int:
        """Number of REAL retrieval attempts.

        The agent loop stores a synthetic placeholder for the
        no-retrieval path. Don't count that against the retrieval total.
        """

        return sum(1 for a in self.attempts if a.query and not a.query.startswith("(none"))


# ----------------------------------------------------------------------------
# NaiveAnswer — baseline output (single retrieval, no agent)
# ----------------------------------------------------------------------------


@dataclass
class NaiveAnswer:
    """The naive RAG baseline. Always one retrieval, never re-queries."""

    user_question: str
    answer_text: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ----------------------------------------------------------------------------
# Token cost estimator — kept here so every module shares the same numbers
# ----------------------------------------------------------------------------


# Claude Haiku 4.5 pricing (USD per 1M tokens, April 2026).
HAIKU_INPUT_PER_M = 1.00
HAIKU_OUTPUT_PER_M = 5.00

# A retrieval call (embed + Chroma query + scoring) adds roughly this many
# tokens of LLM work. The slide uses this constant.
TOKENS_PER_RETRIEVAL_LLM_CALL = 350

# USD-to-INR conversion for the cost-math slide. Approximate.
INR_PER_USD = 83.0


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Pure function. No I/O. The slide and the walkthrough both call this."""

    return (input_tokens / 1_000_000) * HAIKU_INPUT_PER_M + (
        output_tokens / 1_000_000
    ) * HAIKU_OUTPUT_PER_M


def estimate_cost_inr(input_tokens: int, output_tokens: int) -> float:
    return estimate_cost_usd(input_tokens, output_tokens) * INR_PER_USD


def format_currency(usd: float, inr: float) -> str:
    return f"${usd:.4f} USD (~₹{inr:.2f} INR)"
