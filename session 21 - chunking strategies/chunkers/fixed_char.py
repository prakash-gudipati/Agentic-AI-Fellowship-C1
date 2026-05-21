"""
Session 21 — chunkers/fixed_char.py

STRATEGY 1 — fixed-character window with overlap.
This is the S20 baseline. Brutal, simple, breaks sentences in half.
Useful as the floor of the benchmark — every other strategy should beat it.
"""

from typing import List

from .base import Chunk, Chunker


class FixedCharChunker(Chunker):
    name = "fixed_char"

    def __init__(self, size: int = 500, overlap: int = 80) -> None:
        if size <= 0:
            raise ValueError("size must be > 0")
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    def split(self, text: str, source: str, start_id: int = 0) -> List[Chunk]:
        chunks: List[Chunk] = []
        step = self.size - self.overlap
        cursor = 0
        cid = start_id

        while cursor < len(text):
            piece = text[cursor:cursor + self.size].strip()
            if piece:
                chunks.append(Chunk(chunk_id=cid, source=source, text=piece))
                cid += 1
            cursor += step

        return chunks
