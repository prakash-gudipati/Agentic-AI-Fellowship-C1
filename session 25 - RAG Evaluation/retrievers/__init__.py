"""Session 25 — retriever registry.

S25 is an evaluation-focused session. It carries over the retriever
strategy pattern from S22 and the retriever zoo from S23/S24 unchanged.
We deliberately keep only `SimilarityRetriever` in this folder so the
demo stays focused on EVAL, not on retrieval choice.

If you need a retriever from a previous session (BM25, hybrid, MMR,
rerank, parent-child, contextual, HyDE, multi-query, or the S22
FilteredRetriever), import it directly from that session's folder —
every one of them subclasses the same `Retriever` base.
"""

from .base import ChunkIndex, Retrieved, Retriever
from .similarity import SimilarityRetriever

__all__ = ["Retriever", "Retrieved", "ChunkIndex", "SimilarityRetriever"]
