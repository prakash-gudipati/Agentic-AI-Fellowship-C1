"""
Session 20 — chunker.py

STAGE 2 of the RAG pipeline: CHUNK.

We slice each document into overlapping ~500-character pieces. This is the
simplest possible chunker — character-based, fixed window, fixed overlap.
Session 21 covers smarter chunking strategies (sentence-aware, semantic,
recursive). For S20 the goal is to FEEL why chunking matters at all.

Production pattern reinforced:
  - named constants at file top — CHUNK_SIZE / CHUNK_OVERLAP knobs visible
  - every chunk carries its source metadata so retrieval can cite where the
    answer came from
"""

from dataclasses import dataclass
from typing import List, Tuple

# ── Chunking knobs — tune these in the exercise ─────────────────────────────
CHUNK_SIZE = 500       # characters
CHUNK_OVERLAP = 80     # characters of overlap between neighbours


@dataclass
class Chunk:
    """One chunk of source text + its metadata."""
    chunk_id: int
    source: str       # filename of the document this chunk came from
    text: str

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", " ")
        return f"Chunk(id={self.chunk_id}, source={self.source!r}, text={preview!r}...)"


def chunk_text(text: str, source: str, start_id: int = 0,
               size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """Slice `text` into overlapping fixed-size chunks.

    Args:
        text:     the cleaned document text
        source:   the original filename — copied into every chunk
        start_id: numbering offset so multi-document corpora get globally
                  unique chunk IDs
        size:     chunk length in characters
        overlap:  characters of overlap between neighbours
    """
    if size <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks: List[Chunk] = []
    step = size - overlap
    cursor = 0
    cid = start_id

    while cursor < len(text):
        piece = text[cursor:cursor + size].strip()
        if piece:
            chunks.append(Chunk(chunk_id=cid, source=source, text=piece))
            cid += 1
        cursor += step

    return chunks


def chunk_corpus(docs: List[Tuple[str, str]]) -> List[Chunk]:
    """Chunk a list of (source_filename, text) tuples into a single list."""
    all_chunks: List[Chunk] = []
    for source, text in docs:
        chunks = chunk_text(text, source=source, start_id=len(all_chunks))
        print(f"[chunker]  {source:<24}  {len(text):>6} chars  →  {len(chunks):>3} chunks")
        all_chunks.extend(chunks)
    print(f"[chunker]  TOTAL: {len(all_chunks)} chunks across {len(docs)} documents")
    return all_chunks


if __name__ == "__main__":
    # Quick smoke test — run python chunker.py
    sample = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
              "Sed do eiusmod tempor incididunt ut labore et dolore magna "
              "aliqua. Ut enim ad minim veniam, quis nostrud exercitation. " * 6)
    chunks = chunk_text(sample, source="sample.txt")
    for c in chunks:
        print(c)
