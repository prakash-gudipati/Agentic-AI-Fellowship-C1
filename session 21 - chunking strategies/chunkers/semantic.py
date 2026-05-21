"""
Session 21 — chunkers/semantic.py

STRATEGY 4 — semantic chunker.
Embed every sentence. Walk through them. Place a chunk boundary where the
similarity between adjacent sentences DROPS below a threshold (i.e. the topic
just changed). The embeddings find their own boundaries.

This is the most expensive chunker — it calls the embedding API once per
sentence at index-time. Pay once at ingest, not at query.
"""

from typing import List

import numpy as np

from .base import Chunk, Chunker
from .sentence import _split_sentences


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


class SemanticChunker(Chunker):
    name = "semantic"

    def __init__(self, size: int = 800, similarity_drop: float = 0.18) -> None:
        """
        size:             max characters per chunk (hard cap)
        similarity_drop:  open a new chunk when cosine sim between adjacent
                          sentences drops by more than this (in absolute terms)
        """
        self.size = size
        self.threshold = similarity_drop

    def split(self, text: str, source: str, start_id: int = 0) -> List[Chunk]:
        sentences = _split_sentences(text)
        if len(sentences) < 2:
            if not sentences:
                return []
            return [Chunk(chunk_id=start_id, source=source, text=sentences[0])]

        # Lazy import to keep the strategy class free of API-key requirements
        # at import time.
        from embeddings import embed_texts

        vecs = _l2_normalise(embed_texts(sentences))
        # Cosine similarity between adjacent sentences (dot product after L2).
        sims = (vecs[:-1] * vecs[1:]).sum(axis=1)
        # Boundaries — local drops in similarity.
        baseline = float(np.mean(sims))
        boundaries = [i + 1 for i, s in enumerate(sims)
                      if s < baseline - self.threshold]

        chunks: List[Chunk] = []
        cid = start_id
        cursor = 0
        boundaries.append(len(sentences))
        for b in boundaries:
            piece = " ".join(sentences[cursor:b]).strip()
            # Hard size cap — fall back to a sentence-aware split if a semantic
            while len(piece) > self.size:
                head, piece = piece[:self.size], piece[self.size:]
                chunks.append(Chunk(chunk_id=cid, source=source, text=head.strip()))
                cid += 1
            if piece:
                chunks.append(Chunk(chunk_id=cid, source=source, text=piece))
                cid += 1
            cursor = b

        return chunks
