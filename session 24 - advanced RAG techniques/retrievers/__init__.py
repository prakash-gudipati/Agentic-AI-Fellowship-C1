"""Session 23 — retriever registry. The five strategies, behind one interface."""

from .base import Retriever, Retrieved, ChunkIndex
from .similarity import SimilarityRetriever
from .filtered import FilteredRetriever
from .mmr import MMRRetriever
from .bm25 import BM25Retriever
from .hybrid import HybridRetriever
from .rerank import LLMRerankRetriever

# String key → retriever class. The matrix runner iterates this dict.
# Every key must round-trip:  STRATEGIES["bm25"](index) returns a working Retriever.
STRATEGIES = {
    "similarity": SimilarityRetriever,
    "bm25":       BM25Retriever,
    "hybrid":     HybridRetriever,     # default mode = "weighted", α = 0.5
    "mmr":        MMRRetriever,
    "rerank":     LLMRerankRetriever,  # default base = SimilarityRetriever
}

# The 5 retrievers that go into the 5×5 matrix. FilteredRetriever is omitted
# from the matrix because it requires a per-query filter dict — comparing it
# in a fixed grid against unconditional retrievers isn't apples-to-apples.

__all__ = [
    "Retriever", "Retrieved", "ChunkIndex",
    "SimilarityRetriever", "FilteredRetriever", "MMRRetriever",
    "BM25Retriever", "HybridRetriever", "LLMRerankRetriever",
    "STRATEGIES",
]
