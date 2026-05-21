"""Session 22 — retriever registry. The retriever strategy pattern, made literal."""

from .base import Retriever, Retrieved, ChunkIndex
from .similarity import SimilarityRetriever
from .filtered import FilteredRetriever
from .mmr import MMRRetriever

# String key → retriever class. Every key must round-trip:
#   STRATEGIES["mmr"](index) returns a working Retriever.
STRATEGIES = {
    "similarity": SimilarityRetriever,
    "filtered":   FilteredRetriever,
    "mmr":        MMRRetriever,
}

__all__ = [
    "Retriever", "Retrieved", "ChunkIndex",
    "SimilarityRetriever", "FilteredRetriever", "MMRRetriever",
    "STRATEGIES",
]
