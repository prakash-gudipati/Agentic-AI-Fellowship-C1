"""
Session 33 — embeddings.py

A tiny embedding layer with two modes:

  REAL mode   — uses sentence-transformers (all-MiniLM-L6-v2, 384-dim).
                Set USE_REAL_EMBEDDINGS=1 to enable.

  OFFLINE     — a deterministic hash-based 384-dim embedder. Useful for
                CI smoke tests and for classrooms without internet. Same
                trick as S32's HashEmbedder, scaled up so it cohabits the
                same Chroma collection if a student flips the env var
                mid-build.

Why an offline path? S19 introduced ChromaDB and sentence-transformers,
but downloading the model on session day kills demos. A deterministic
hash embedder lets the demo run end-to-end with `pip install chromadb`
alone.

The two embedders disagree about which chunks are closest to which
queries — that is by design. The hash embedder is good enough to show
the AGENTIC LOOP working, not to show retrieval QUALITY. The walkthrough
script flags this and recommends students set USE_REAL_EMBEDDINGS=1 for
their exercise.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import List, Sequence


EMBED_DIM = 384


# ----------------------------------------------------------------------------
# Hash embedder — deterministic, offline, no model download
# ----------------------------------------------------------------------------


def _hash_embed(text: str, dim: int = EMBED_DIM) -> List[float]:
    """Map text to a 384-d unit vector.

    Token-bag hashed into a fixed-dim float vector, then L2-normalised.
    Good enough that "what is the refund window" and "refund policy" end
    up neighbours, which is all we need for the agentic loop demos.
    """

    vec = [0.0] * dim
    # Tokenise on whitespace + lowercase. Drop short stop-y tokens.
    tokens = [t for t in text.lower().split() if len(t) > 2]
    for tok in tokens:
        h = hashlib.sha1(tok.encode("utf-8")).digest()
        # Spread each token across 4 slots using different byte windows
        # so synonym overlap is more forgiving.
        for offset in (0, 4, 8, 12):
            idx = int.from_bytes(h[offset : offset + 4], "big") % dim
            sign = 1.0 if h[offset] % 2 == 0 else -1.0
            vec[idx] += sign

    # L2 normalise so cosine similarity works.
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


# ----------------------------------------------------------------------------
# Real embedder — sentence-transformers
# ----------------------------------------------------------------------------


class _RealEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Import lazily so the offline path never pays the import cost.
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self._model.encode(list(texts))]


# Lazy singleton — instantiated on first use only.
_real: _RealEmbedder | None = None


# ----------------------------------------------------------------------------
# Public API — embed and embed_many
# ----------------------------------------------------------------------------


def _use_real() -> bool:
    return os.environ.get("USE_REAL_EMBEDDINGS", "") == "1"


def embed(text: str) -> List[float]:
    """Embed a single text. Routes to real or hash based on env var."""

    if _use_real():
        global _real
        if _real is None:
            _real = _RealEmbedder()
        return _real.embed([text])[0]
    return _hash_embed(text)


def embed_many(texts: Sequence[str]) -> List[List[float]]:
    """Embed a batch of texts."""

    if _use_real():
        global _real
        if _real is None:
            _real = _RealEmbedder()
        return _real.embed(list(texts))
    return [_hash_embed(t) for t in texts]


def embedder_label() -> str:
    """Short string used by the trace logger to mark which embedder is live."""

    return "sentence-transformers" if _use_real() else "hash-embedder (offline)"
