"""
Session 33 — quality_gate.py

PROD PATTERN: Retrieval Quality Gate.

Every batch of retrieved chunks is scored 1..5 against the query BEFORE
the agent uses them. If the average relevance is below a threshold (3.5
by default), the gate flags the batch as failed and the agent re-queries
with a different formulation rather than answering from weak evidence.

This is the lead pattern the curriculum names for S33. The point is
production teams don't generate from whatever the vector store happens
to return — they put a relevance gate in front of generation.

Two modes, same interface:

  score_chunks(query, chunks, llm) -> List[int]
      Calls the LLM to rate each chunk 1..5.

  score_chunks_heuristic(query, chunks) -> List[int]
      Cheap token-overlap fallback. Used by demos and tests so the
      gate runs even when the LLM is unavailable. Also used by
      LLMClient's FAKE_LLM path under the hood.

The gate then bundles those scores into a Verdict object the agent
reads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from llm_client import LLMClient
from prompts import QUALITY_GATE_SYSTEM
from rag_types import RetrievedChunk


# Production teams tune this threshold per domain. 3.5 is the default
# from the S33 curriculum spec.
DEFAULT_PASS_THRESHOLD = 3.5


# ----------------------------------------------------------------------------
# Verdict — what gate.evaluate() returns
# ----------------------------------------------------------------------------


@dataclass
class GateVerdict:
    average_relevance: float
    threshold: float
    passed: bool
    reason: str

    def short(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return (
            f"gate: {flag}  avg={self.average_relevance:.2f}  "
            f"threshold={self.threshold:.2f}  ({self.reason})"
        )


# ----------------------------------------------------------------------------
# Scorers
# ----------------------------------------------------------------------------


def score_chunks(
    query: str,
    chunks: List[RetrievedChunk],
    llm: Optional[LLMClient] = None,
) -> List[int]:
    """LLM-based scorer. Mutates chunks in place AND returns the score list."""

    if not chunks:
        return []
    llm = llm or LLMClient()

    chunk_lines = []
    for c in chunks:
        # Single-line text so the parser is unambiguous.
        flat = c.text.replace("\n", " ").strip()
        chunk_lines.append(
            f"- chunk_id={c.chunk_id} source={c.source} text={flat}"
        )
    user_prompt = (
        "QUERY: "
        f"{query}\n\n"
        "CHUNKS:\n" + "\n".join(chunk_lines) + "\n"
    )
    raw = llm.complete(QUALITY_GATE_SYSTEM, user_prompt, max_tokens=600)
    scores_by_id = _parse_score_lines(raw)

    out: List[int] = []
    for c in chunks:
        score = scores_by_id.get(c.chunk_id)
        if score is None:
            # If the LLM forgot a chunk, fall back to the heuristic for
            # just that one chunk so we never have a missing score.
            score = score_chunks_heuristic(query, [c])[0]
        score = max(1, min(5, int(score)))
        c.relevance = score
        out.append(score)
    return out


def score_chunks_heuristic(
    query: str, chunks: List[RetrievedChunk]
) -> List[int]:
    """Cheap fallback: token overlap with a small source-aware boost."""

    q_tokens = _keywords(query)
    out: List[int] = []
    for c in chunks:
        c_tokens = _keywords(c.text)
        overlap = q_tokens & c_tokens
        score = 1
        if overlap:
            score = min(5, 1 + len(overlap))
        src_token = c.source.split("_", 1)[-1].split(".", 1)[0]
        if src_token in q_tokens:
            score = min(5, score + 1)
        if len(c.text) < 80:
            score = max(1, score - 1)
        c.relevance = score
        out.append(score)
    return out


# ----------------------------------------------------------------------------
# Gate
# ----------------------------------------------------------------------------


def evaluate(
    query: str,
    chunks: List[RetrievedChunk],
    llm: Optional[LLMClient] = None,
    threshold: float = DEFAULT_PASS_THRESHOLD,
    use_llm: bool = True,
) -> GateVerdict:
    """Score the chunks and return a pass/fail verdict.

    Default mode uses the LLM. Set use_llm=False for the heuristic
    fallback (cheaper, used by the budget-exhausted branch where we
    don't want to spend more on scoring).
    """

    if not chunks:
        return GateVerdict(
            average_relevance=0.0,
            threshold=threshold,
            passed=False,
            reason="no chunks returned",
        )
    if use_llm:
        scores = score_chunks(query, chunks, llm=llm)
    else:
        scores = score_chunks_heuristic(query, chunks)
    avg = sum(scores) / len(scores)
    passed = avg >= threshold
    reason = (
        f"avg relevance {avg:.2f} {'≥' if passed else '<'} threshold "
        f"{threshold:.2f}"
    )
    return GateVerdict(
        average_relevance=avg,
        threshold=threshold,
        passed=passed,
        reason=reason,
    )


# ----------------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------------


_SCORE_LINE_RE = re.compile(
    r"SCORE\s+chunk_id=(\S+)\s+relevance=(\d)",
    re.IGNORECASE,
)


def _parse_score_lines(text: str) -> dict:
    """Return a {chunk_id: int_score} map from the gate's stdout."""

    out: dict = {}
    for m in _SCORE_LINE_RE.finditer(text):
        cid = m.group(1)
        try:
            s = int(m.group(2))
        except ValueError:
            continue
        out[cid] = max(1, min(5, s))
    return out


def _keywords(text: str) -> set:
    out = set()
    for tok in re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower()):
        if len(tok) > 3:
            out.add(tok)
    return out
