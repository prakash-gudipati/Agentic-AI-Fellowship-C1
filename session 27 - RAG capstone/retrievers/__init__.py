"""Session 27 -- retrievers package.

Three retrieval strategies, all reading from the SAME `ChunkIndex`
(either the in-memory one from S22 or the ChromaDB-backed one
introduced in S27).

  SimilarityRetriever  -- pure top-K cosine.
  MMRRetriever          -- Maximal Marginal Relevance + lambda knob.
  HybridRetriever       -- dense + BM25 blend, alpha knob.

The capstone's `comparison_harness.py` runs all three against the
same golden set and ranks them by composite Ragas score. The student
ships the winner.
"""

from .base import ChunkIndex, Retriever, Retrieved
from .similarity import SimilarityRetriever
from .mmr import MMRRetriever
from .hybrid import HybridRetriever

__all__ = [
    "ChunkIndex",
    "Retriever",
    "Retrieved",
    "SimilarityRetriever",
    "MMRRetriever",
    "HybridRetriever",
]
