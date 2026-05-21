"""
Session 21 — chunkers/sentence.py

STRATEGY 2 — sentence-aware chunker.
Pack as many whole sentences as possible into each chunk without exceeding
`size` characters. Sentences NEVER get cut in half.

Tries NLTK's sentence tokeniser first; falls back to a regex if NLTK data
isn't installed (so the demo never blocks on a download).
"""

import re
from typing import List

from .base import Chunk, Chunker


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences. NLTK if available, regex fallback otherwise."""
    try:
        from nltk.tokenize import sent_tokenize  # type: ignore
        return sent_tokenize(text)
    except (ImportError, LookupError):
        # Regex fallback — handles ., !, ?, but not "Mr." style edge cases.
        # Acceptable for the demo. Production code uses spaCy or a custom rule.
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]


class SentenceChunker(Chunker):
    name = "sentence"

    def __init__(self, size: int = 500) -> None:
        self.size = size

    def split(self, text: str, source: str, start_id: int = 0) -> List[Chunk]:
        sentences = _split_sentences(text)
        chunks: List[Chunk] = []
        cid = start_id
        buf: List[str] = []
        buf_len = 0

        for s in sentences:
            sl = len(s) + 1  # +1 for the joining space
            if buf and buf_len + sl > self.size:
                chunks.append(Chunk(chunk_id=cid, source=source,
                                    text=" ".join(buf).strip()))
                cid += 1
                buf, buf_len = [], 0
            buf.append(s)
            buf_len += sl

        if buf:
            chunks.append(Chunk(chunk_id=cid, source=source,
                                text=" ".join(buf).strip()))
        return chunks
