"""Session 21 — chunker registry. The strategy pattern, made literal."""

from .base import Chunk, Chunker
from .fixed_char import FixedCharChunker
from .sentence import SentenceChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .structure import StructureChunker

# String key → instantiator. Every key must round-trip:
#   STRATEGIES["recursive"]() returns a working Chunker.
STRATEGIES = {
    "fixed_char": FixedCharChunker,
    "sentence":   SentenceChunker,
    "recursive":  RecursiveChunker,
    "semantic":   SemanticChunker,
    "structure":  StructureChunker,
}

__all__ = [
    "Chunk", "Chunker",
    "FixedCharChunker", "SentenceChunker", "RecursiveChunker",
    "SemanticChunker", "StructureChunker",
    "STRATEGIES",
]
