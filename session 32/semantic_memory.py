"""
Session 32 — semantic_memory.py

PROD PATTERN — Semantic Memory via Vector Recall.

This is the LONG-TERM, SEARCHABLE memory layer. Every turn that leaves the
working-memory buffer is not just summarised (summariser.py) — its raw
text is also embedded and pushed into a tiny vector store. The agent can
then RETRIEVE relevant past turns when the current question hints that
something earlier in the session matters.

We deliberately reuse the Phase 4 mental model:
  embed -> store -> retrieve by cosine similarity -> top-K.

But we keep the implementation tiny — a pure-numpy index over a fake but
deterministic 64-dim hash-embedding by default, with a clean opt-in path
to real Anthropic / OpenAI embeddings when the env var is set. Students
who completed S20-S27 will recognise this as the exact pattern they built,
re-wired to operate on the agent's OWN conversation history rather than a
document corpus. Same machine. New corpus.

What's named here:
  - HashEmbedder         — offline fallback; deterministic 64-d signed
                            embedding from a SHA-256 of the input.
  - SemanticMemory.add(turn)
  - SemanticMemory.recall(query, k, min_score) -> list[SemanticHit]
  - SemanticMemory.size()  -> int
  - SemanticMemory.turn_ids() -> list[str]

Why the hash fallback?
  - The fellowship classroom is global. Embedding-model SDKs add a
    network dependency, an API key, and a cost. The hash embedder is
    deterministic, free, offline, and good enough to demonstrate the
    PATTERN. Real production code swaps it for OpenAI text-embedding-3 or
    voyage-3 — that swap is one method body, not a redesign.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import List, Optional, Protocol

from memory_types import SemanticHit, Turn


# ----------------------------------------------------------------------------
# Embedder interface
# ----------------------------------------------------------------------------


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...
    def dim(self) -> int: ...


# ----------------------------------------------------------------------------
# HashEmbedder — deterministic offline fallback
# ----------------------------------------------------------------------------


class HashEmbedder:
    """
    A 64-d signed embedding produced from SHA-256 chunks of the input.

    Properties we care about:
      - Deterministic across machines, runs, days.
      - Cheap (no API call).
      - L2-normalised so cosine similarity reduces to dot product.

    Properties we do NOT have:
      - Real semantic similarity. Two paraphrases will NOT be close.
      - Robustness to typos.

    The hash embedder is here to demonstrate the PATTERN — the rest of the
    code is identical when you swap in a real embedder. We make this
    limitation visible in the demo by querying with exact substrings of
    earlier turns, not paraphrases.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self._dim = dimensions

    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        # Build enough hash bytes to fill the dim, two bytes per dimension.
        needed_bytes = self._dim * 2
        accum = b""
        seed = text.encode("utf-8", errors="replace")
        counter = 0
        while len(accum) < needed_bytes:
            accum += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            counter += 1

        # Map every pair of bytes into a centered float in [-1, 1].
        raw = []
        for i in range(self._dim):
            lo = accum[2 * i]
            hi = accum[2 * i + 1]
            v = (hi * 256 + lo) / 65535.0  # in [0, 1]
            raw.append(v * 2.0 - 1.0)      # center to [-1, 1]

        # L2 normalise so cosine == dot product.
        norm = math.sqrt(sum(x * x for x in raw))
        if norm == 0:
            return raw
        return [x / norm for x in raw]


# ----------------------------------------------------------------------------
# Cosine similarity (on pre-normalised vectors == dot product)
# ----------------------------------------------------------------------------


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ----------------------------------------------------------------------------
# SemanticMemory — tiny vector index
# ----------------------------------------------------------------------------


class SemanticMemory:
    """
    Append-only vector index over the agent's past turns.

    Implementation is deliberately the simplest thing that works:
      - One list of turns.
      - One parallel list of L2-normalised embeddings.
      - Recall computes dot products and returns top-K above a threshold.

    Production code swaps this for ChromaDB / FAISS / pgvector. The shape
    of the public API is the same on purpose — students should feel the
    callback to Session 19 the second they read recall().
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.embedder = embedder or HashEmbedder()
        self._turns: List[Turn] = []
        self._vectors: List[List[float]] = []

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, turn: Turn) -> None:
        """
        Embed and store one turn.

        Idempotent on turn_id — if we have already added this turn, the
        call is a no-op. This matters because the agent loop calls add()
        every time a turn is evicted from working memory, and a replay
        scenario can re-evict the same logical turn.
        """

        if any(t.turn_id == turn.turn_id for t in self._turns):
            return

        vec = self.embedder.embed(turn.content)
        self._turns.append(turn)
        self._vectors.append(vec)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.15,
    ) -> List[SemanticHit]:
        """
        Return up to k turns whose embedding is closest to the query.

        The min_score floor matters in production — a vector store
        always returns SOMETHING, even when nothing relevant exists.
        Without a threshold the agent ends up injecting unrelated old
        turns into the prompt and confusing itself.
        """

        if not self._turns:
            return []

        q_vec = self.embedder.embed(query)
        scored = [
            SemanticHit(turn=t, score=_dot(q_vec, v))
            for t, v in zip(self._turns, self._vectors)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return [h for h in scored[:k] if h.score >= min_score]

    def size(self) -> int:
        return len(self._turns)

    def turn_ids(self) -> List[str]:
        return [t.turn_id for t in self._turns]


# ----------------------------------------------------------------------------
# Factory helper — picks the embedder based on env vars
# ----------------------------------------------------------------------------


def default_embedder() -> Embedder:
    """
    Return HashEmbedder unless USE_REAL_EMBEDDINGS=1 is set.

    The real-embeddings path is intentionally not implemented in this
    teaching build — students see the env-var hook and read the comment.
    The S19 / S20-22 sessions already covered the swap; this session is
    about the PATTERN, not the embedding model choice.
    """

    if os.environ.get("USE_REAL_EMBEDDINGS", "") == "1":
        raise NotImplementedError(
            "USE_REAL_EMBEDDINGS=1 is reserved for the production extension "
            "exercise in Session_32_Exercise.docx. Drop your OpenAI / Voyage "
            "client here and you have a production semantic-memory layer."
        )
    return HashEmbedder()
