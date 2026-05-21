"""
Session 23 / 27 -- retrievers/hybrid.py

STRATEGY 3 -- Hybrid Dense + Sparse retrieval.

WHY THIS RETRIEVER EXISTS
-------------------------
Dense retrievers (cosine on embeddings) understand SEMANTICS but can
miss exact-keyword matches -- "ChromaDB" vs "Chroma DB" can map to
slightly different vectors. Sparse retrievers (BM25, TF-IDF) excel at
EXACT TERM matches but are blind to paraphrase.

Hybrid retrieval blends both. We compute:
  - dense_score   = cosine(query, chunk) via the L2-normalised matrix
  - sparse_score  = BM25 score for the query against the chunk's text
and combine via min-max normalisation:
  blended = alpha * dense_norm + (1 - alpha) * sparse_norm

The alpha knob:
  alpha = 1.0  -> pure dense (same as SimilarityRetriever)
  alpha = 0.0  -> pure sparse (lexical only)
  alpha = 0.5  -> the production sweet spot in most papers / blogs

BM25 IMPLEMENTATION
-------------------
Inlined for zero external dependencies. The classic Robertson-Sparck
Jones BM25 with k1=1.5, b=0.75. We tokenise on whitespace + lowercase;
production systems use heavier tokenisers (Lucene-style), but the
shape of the algorithm is identical.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

from .base import Retriever, Retrieved


_TOKEN_RE = re.compile(r"\w+")
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenise(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class HybridRetriever(Retriever):
    """Dense + BM25 blend.

    Args:
        alpha: dense vs sparse weight (1.0 = pure dense, 0.0 = pure BM25).
    """

    name = "hybrid"

    def __init__(self, index, *, alpha: float = 0.5) -> None:
        super().__init__(index)
        self.alpha = float(alpha)
        self._bm25_ready = False
        self._doc_tokens: List[List[str]] = []
        self._doc_lens: np.ndarray = np.zeros(0)
        self._avg_len: float = 0.0
        self._idf: Dict[str, float] = {}

    # ── BM25 pre-compute ──────────────────────────────────────────────

    def _prepare_bm25(self) -> None:
        if self._bm25_ready and len(self._doc_tokens) == len(self.index):
            return
        self._doc_tokens = [_tokenise(c.text) for c in self.index.chunks]
        self._doc_lens = np.array(
            [len(t) for t in self._doc_tokens], dtype=np.float32)
        self._avg_len = (
            float(self._doc_lens.mean()) if self._doc_lens.size else 1.0
        )
        n_docs = max(len(self._doc_tokens), 1)
        df: Counter = Counter()
        for toks in self._doc_tokens:
            for term in set(toks):
                df[term] += 1
        self._idf = {
            term: math.log(1.0 + (n_docs - dfi + 0.5) / (dfi + 0.5))
            for term, dfi in df.items()
        }
        self._bm25_ready = True

    def _bm25_scores(self, query: str) -> np.ndarray:
        self._prepare_bm25()
        q_terms = _tokenise(query)
        scores = np.zeros(len(self._doc_tokens), dtype=np.float32)
        for i, doc in enumerate(self._doc_tokens):
            if not doc:
                continue
            tf = Counter(doc)
            dl = self._doc_lens[i]
            norm = (1.0 - _BM25_B + _BM25_B * (dl / max(self._avg_len, 1.0)))
            s = 0.0
            for term in q_terms:
                idf = self._idf.get(term)
                if idf is None:
                    continue
                tf_term = tf.get(term, 0)
                if tf_term == 0:
                    continue
                s += idf * ((tf_term * (_BM25_K1 + 1.0))
                            / (tf_term + _BM25_K1 * norm))
            scores[i] = s
        return scores

    # ── Pick ──────────────────────────────────────────────────────────

    def pick(self, query: str, k: int = 3,
             **kwargs: Any) -> List[Retrieved]:
        if len(self.index) == 0:
            return []

        q_vec = self._embed_query(query)
        dense = self._cosine_scores(q_vec)
        sparse = self._bm25_scores(query)

        # Min-max normalise both score vectors to [0, 1] before blend.
        dense_norm = _minmax(dense)
        sparse_norm = _minmax(sparse)
        blended = self.alpha * dense_norm + (1.0 - self.alpha) * sparse_norm

        top = np.argsort(-blended)[:k]
        results = [Retrieved(chunk=self.index.chunks[i],
                              score=float(blended[i]))
                    for i in top]
        print(f"[hybrid] query={query!r}  alpha={self.alpha:.2f}  "
              f"top-{k} blended: "
              + ", ".join(f"{r.score:.3f}" for r in results))
        return results


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)
