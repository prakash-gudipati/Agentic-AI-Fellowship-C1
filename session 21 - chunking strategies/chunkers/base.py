"""
Session 21 — chunkers/base.py

Common interface for every chunking strategy. Production pattern — STRATEGY
PATTERN. Every chunker exposes the same `(text, source) → List[Chunk]` shape,
so the rest of the RAG pipeline doesn't care which one is in use. Swap the
implementation, keep the contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Chunk:
    """One chunk of source text + its provenance metadata.

    Identical shape to the S20 Chunk so retriever.py keeps working unchanged.
    """
    chunk_id: int
    source: str       # filename of the document this chunk came from
    text: str

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", " ")
        return f"Chunk(id={self.chunk_id}, source={self.source!r}, text={preview!r}...)"


class Chunker(ABC):
    """Abstract chunker. Every strategy subclasses this and implements split()."""

    name: str = "abstract"

    @abstractmethod
    def split(self, text: str, source: str, start_id: int = 0) -> List[Chunk]:
        """Slice `text` into chunks. Must populate chunk_id starting at start_id."""
        raise NotImplementedError

    def chunk_corpus(self, docs: List[Tuple[str, str]]) -> List[Chunk]:
        """Chunk a list of (source_filename, text) tuples into one flat list.

        Provided for convenience — every strategy gets this for free.
        """
        all_chunks: List[Chunk] = []
        for source, text in docs:
            chunks = self.split(text, source=source, start_id=len(all_chunks))
            print(f"[{self.name:<10}]  {source:<24}  "
                  f"{len(text):>6} chars  →  {len(chunks):>3} chunks")
            all_chunks.extend(chunks)
        return all_chunks
