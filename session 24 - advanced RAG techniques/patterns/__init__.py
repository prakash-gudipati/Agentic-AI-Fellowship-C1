"""Session 24 - advanced RAG patterns package.

Every module here is a DECORATOR that wraps the base RAG system from S20-S23.
None of them replace the existing retrievers - they compose on top.

Patterns shipped today:
  - parent_child.ParentChildRetriever  - small to match, big to read
  - contextual.ContextualChunker       - prepend per-chunk context before embedding
  - hyde.HyDERetriever                 - hypothetical document embeddings
  - multi_query.MultiQueryRetriever    - generate N variants, fuse with RRF
"""

from .parent_child import ParentChildRetriever
from .contextual import ContextualChunker, build_context_for_chunk
from .hyde import HyDERetriever
from .multi_query import MultiQueryRetriever

__all__ = [
    "ParentChildRetriever",
    "ContextualChunker", "build_context_for_chunk",
    "HyDERetriever",
    "MultiQueryRetriever",
]
