"""
Session 21 — chunkers/structure.py

STRATEGY 5 — document-structure chunker.
The author already broke the document into sections — chapters, headings,
Markdown #, HTML <h1>/<h2>. Honour those boundaries. Each top-level section
becomes one chunk.

This is the right answer for technical documentation, legal contracts, and
anything with a strong heading hierarchy.

For the S20 corpus this works beautifully on the WidgetMax product manual
(Chapter 1, Chapter 2, ...) and the Intro-to-RAG paper (1. The Problem,
2. The Five Stages, ...). For arbitrary blog posts it falls back to
paragraph splits.
"""

import re
from typing import List

from .base import Chunk, Chunker

# Patterns that look like document headings (in order of confidence).
HEADING_PATTERNS = [
    re.compile(r"^\s*Chapter\s+\d+\s*[—–-].*$", re.M),
    re.compile(r"^\s*Section\s+\d+\s*[—–-].*$", re.M),
    re.compile(r"^\s*\d+\.\s+[A-Z].{2,}$", re.M),       # "1. The Problem RAG Solves"
    re.compile(r"^\s*#{1,3}\s+.+$", re.M),                # "# heading", "## heading"
    re.compile(r"^\s*[A-Z][A-Z0-9 ]{4,}$", re.M),         # ALL-CAPS line
]


class StructureChunker(Chunker):
    name = "structure"

    def __init__(self, max_size: int = 2000) -> None:
        # If a section is huge, split it further on paragraph boundaries.
        self.max_size = max_size

    def split(self, text: str, source: str, start_id: int = 0) -> List[Chunk]:
        # Find all heading positions across all patterns.
        positions: List[int] = []
        for pat in HEADING_PATTERNS:
            positions.extend(m.start() for m in pat.finditer(text))
        positions = sorted(set(positions))

        # If we found no headings, fall back to paragraph splits.
        if not positions:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            return [Chunk(chunk_id=start_id + i, source=source, text=p)
                    for i, p in enumerate(paragraphs)]

        # Build sections — from each heading to the next.
        positions.append(len(text))
        chunks: List[Chunk] = []
        cid = start_id
        for i in range(len(positions) - 1):
            section = text[positions[i]:positions[i + 1]].strip()
            if not section:
                continue
            # If section is over max_size, split it on paragraph boundaries.
            if len(section) > self.max_size:
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section)
                              if p.strip()]
                for p in paragraphs:
                    chunks.append(Chunk(chunk_id=cid, source=source, text=p))
                    cid += 1
            else:
                chunks.append(Chunk(chunk_id=cid, source=source, text=section))
                cid += 1
        return chunks
