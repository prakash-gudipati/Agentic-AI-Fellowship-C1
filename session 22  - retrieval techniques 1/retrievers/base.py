"""
Session 22 — retrievers/base.py

THE RETRIEVER STRATEGY PATTERN.

Every retriever exposes the same `(query, k, **kwargs) → List[Retrieved]`
interface. The rest of the RAG pipeline doesn't care which strategy is in
use — swap the implementation, keep the contract.

This file also defines `ChunkIndex` — the shared store of chunks + their
embedding vectors. All three retrievers read from the SAME ChunkIndex,
because the embedding step shouldn't be repeated three times. That's the
whole point of decoupling indexing from retrieval.

Production patterns introduced:
  - retriever strategy pattern (NEW in S22)
  - one index, many retrievers (the right factoring for swapping strategies)
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# Add the parent dir to sys.path so `chunker` and `embeddings` resolve when
# this file is imported as `retrievers.base`.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from chunker import Chunk
from embeddings import embed_texts, embed_one, l2_normalise


@dataclass
class Retrieved:
    """One retrieved chunk plus the score the retriever assigned to it."""
    chunk: Chunk
    score: float

    def __repr__(self) -> str:
        preview = self.chunk.text[:60].replace("\n", " ")
        return (f"Retrieved(score={self.score:.3f}, "
                f"source={self.chunk.source!r}, "
                f"meta={self.chunk.metadata}, text={preview!r})")


class ChunkIndex:
    """Shared storage layer — all retrievers read from this.

    Holds the chunks themselves and a parallel matrix of L2-normalised
    embedding vectors so every retriever can do cosine similarity as a
    single dot product.
    """

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.vectors: np.ndarray = np.zeros((0, 1536), dtype=np.float32)

    def add(self, chunks: List[Chunk]) -> None:
        """Embed the given chunks and append them. Vectors are L2-normalised."""
        if not chunks:
            return
        new = embed_texts([c.text for c in chunks])
        new = l2_normalise(new)
        self.chunks.extend(chunks)
        self.vectors = (new if self.vectors.shape[0] == 0
                        else np.vstack([self.vectors, new]))
        print(f"[index] holds {len(self.chunks)} chunks  "
              f"(matrix shape={self.vectors.shape})")

    def __len__(self) -> int:
        return len(self.chunks)


# ── Retriever interface ─────────────────────────────────────────────────────

class Retriever(ABC):
    """Every retriever subclasses this. Implements pick()."""

    name: str = "abstract"

    def __init__(self, index: ChunkIndex) -> None:
        self.index = index

    @abstractmethod
    def pick(self, query: str, k: int = 3,
             **kwargs: Any) -> List[Retrieved]:
        """Return up to `k` retrieved chunks for `query`."""
        raise NotImplementedError

    # ── Helpers shared by every retriever ───────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed `query` and L2-normalise. Returns a (1536,) vector."""
        q = embed_one(query)
        return l2_normalise(q)

    def _cosine_scores(self, q_vec: np.ndarray) -> np.ndarray:
        """Cosine similarity of `q_vec` against every chunk in the index.
        Both sides are L2-normalised, so cosine == dot product.
        """
        if len(self.index) == 0:
            return np.zeros(0, dtype=np.float32)
        return self.index.vectors @ q_vec
