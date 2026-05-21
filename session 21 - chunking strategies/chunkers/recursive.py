"""
Session 21 — chunkers/recursive.py

STRATEGY 3 — recursive chunker. THE LANGCHAIN DEFAULT.
Try to split on the biggest separator first (paragraph). If a chunk is still
too big, recurse — try sentence boundaries. Still too big? Try word boundaries.
Last resort — character window (the S20 baseline).

This is the most-used chunker in production RAG. Knowing how it works under
the hood is non-negotiable before students touch LangChain in S37.
"""

import re
from typing import List

from .base import Chunk, Chunker

# Order of separators to try. Each step is "smaller" than the last.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(Chunker):
    name = "recursive"

    def __init__(self, size: int = 500, overlap: int = 80,
                 separators: List[str] = None) -> None:
        self.size = size
        self.overlap = overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def split(self, text: str, source: str, start_id: int = 0) -> List[Chunk]:
        pieces = self._recursive_split(text, self.separators)
        chunks: List[Chunk] = []
        cid = start_id
        # Add overlap by tail-joining adjacent small pieces.
        joined = self._merge_small(pieces, self.size, self.overlap)
        for p in joined:
            p = p.strip()
            if p:
                chunks.append(Chunk(chunk_id=cid, source=source, text=p))
                cid += 1
        return chunks

    def _recursive_split(self, text: str, seps: List[str]) -> List[str]:
        """Split `text` on the first separator; recurse on any over-size pieces."""
        if len(text) <= self.size:
            return [text]
        if not seps:
            # Hard fallback — slice by characters.
            return [text[i:i + self.size]
                    for i in range(0, len(text), self.size)]

        sep, rest = seps[0], seps[1:]
        if sep == "":
            return [text[i:i + self.size]
                    for i in range(0, len(text), self.size)]

        parts = text.split(sep)
        out: List[str] = []
        for p in parts:
            piece = p + (sep if sep.strip() else "")
            if len(piece) <= self.size:
                out.append(piece)
            else:
                out.extend(self._recursive_split(piece, rest))
        return out

    def _merge_small(self, pieces: List[str], size: int, overlap: int) -> List[str]:
        """Greedily concatenate adjacent small pieces, with overlap on the boundary."""
        merged: List[str] = []
        buf = ""
        for p in pieces:
            if not buf:
                buf = p
                continue
            if len(buf) + len(p) <= size:
                buf += p
            else:
                merged.append(buf)
                # tail-overlap: prepend the last `overlap` chars of buf to the next chunk
                tail = buf[-overlap:] if overlap > 0 else ""
                buf = (tail + p).strip()
        if buf:
            merged.append(buf)
        return merged
