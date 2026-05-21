"""
Session 22 — chunker.py

A slim wrapper around the S21 RECURSIVE chunker (the LangChain default), with
two things added today:

  1. Each Chunk now carries `metadata` copied from its parent Doc, so any
     downstream retriever can filter by it.
  2. The Chunk shape is shared with retrievers — same dataclass, same fields.

We use the recursive strategy as the default because S21's benchmark showed
it to be a strong baseline across all three corpus types. Tuning the chunker
itself is not today's lesson — today's lesson is the retriever.

Production patterns reinforced:
  - strategy pattern (S21) — chunker is interchangeable
  - metadata as an ingest-time obligation (S22)
  - pipeline logging with [chunker] prefix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from loaders import Doc

# Try the S21 recursive chunker. If the path isn't set up, fall back to a
# minimal in-file recursive chunker so this module stands alone.
#
# IMPORTANT: we APPEND S21 to sys.path (not insert at front). If S21 went to
# position 0, Python would pick up S21's older embeddings.py — which is
# missing the l2_normalise helper that S22 added — before our local one.
# Appending keeps Session_22/Code/ as the priority for any module name we
# share with S21 (embeddings.py, loaders.py).
try:
    import os
    import sys
    S21 = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "Session_21", "Code"))
    if os.path.isdir(S21) and S21 not in sys.path:
        sys.path.append(S21)
    from chunkers.recursive import RecursiveChunker  # type: ignore
    _SOURCE = "S21.recursive"
except Exception:
    RecursiveChunker = None  # type: ignore
    _SOURCE = "S22.fallback"


# ── Public Chunk type ───────────────────────────────────────────────────────

@dataclass
class Chunk:
    """One chunk of text plus its provenance and metadata.

    Same shape as the S21 Chunk, with a metadata dict added.
    """
    chunk_id: int
    source: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", " ")
        return (f"Chunk(id={self.chunk_id}, source={self.source!r}, "
                f"meta={self.metadata}, text={preview!r}...)")


# ── Default chunker config ──────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80


def chunk_corpus(docs: List[Doc],
                 size: int = DEFAULT_CHUNK_SIZE,
                 overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Chunk]:
    """Recursive-chunk every doc and copy its metadata onto each chunk."""
    if RecursiveChunker is not None:
        return _chunk_with_s21(docs, size, overlap)
    return _chunk_with_fallback(docs, size, overlap)


def _chunk_with_s21(docs: List[Doc], size: int, overlap: int) -> List[Chunk]:
    """Use the S21 recursive chunker, then attach metadata to each chunk."""
    chunker = RecursiveChunker(size=size, overlap=overlap)
    out: List[Chunk] = []
    cid = 0
    for d in docs:
        s21_chunks = chunker.split(d.text, source=d.source, start_id=cid)
        for sc in s21_chunks:
            out.append(Chunk(
                chunk_id=sc.chunk_id,
                source=sc.source,
                text=sc.text,
                metadata=dict(d.metadata),  # copy so chunks don't share refs
            ))
        cid += len(s21_chunks)
        print(f"[chunker] {d.source:<24}  →  {len(s21_chunks):>3} chunks "
              f"(via {_SOURCE})")
    print(f"[chunker] total chunks: {len(out)}")
    return out


def _chunk_with_fallback(docs: List[Doc], size: int, overlap: int) -> List[Chunk]:
    """Tiny built-in recursive chunker. Used only when S21 is not on the path."""
    out: List[Chunk] = []
    cid = 0
    for d in docs:
        text = d.text
        pieces: List[str] = []
        # Try paragraph splits, fall back to character window.
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para) <= size:
                pieces.append(para)
            else:
                for i in range(0, len(para), size - overlap):
                    pieces.append(para[i:i + size])
        for p in pieces:
            out.append(Chunk(
                chunk_id=cid, source=d.source, text=p,
                metadata=dict(d.metadata),
            ))
            cid += 1
        print(f"[chunker] {d.source:<24}  →  {len(pieces):>3} chunks "
              f"(via {_SOURCE})")
    print(f"[chunker] total chunks: {len(out)}")
    return out


if __name__ == "__main__":
    import os
    from loaders import load_corpus
    here = os.path.dirname(__file__)
    docs = load_corpus([
        os.path.join(here, "data", "intro_to_rag.pdf"),
        os.path.join(here, "data", "product_manual.pdf"),
        os.path.join(here, "data", "sample_article.html"),
    ])
    chunks = chunk_corpus(docs)
    print()
    for c in chunks[:3]:
        print(c)
