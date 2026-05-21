"""
Session 23 — retrievers/hybrid.py

STRATEGY 5 — HYBRID search. Two scoring signals, one result list.

Dense (cosine) and sparse (BM25) fail in opposite directions:
  - Dense misses exact terms (product codes, error numbers, person names)
  - Sparse misses synonyms ("car" vs "automobile")

Hybrid combines them. Two fusion methods are implemented here:

  MODE = "weighted"   — α-fusion after min-max normalisation
                        final = α · dense_norm  +  (1 − α) · sparse_norm
                        α is the new knob. 0.5 = balanced. Higher = lean dense.

  MODE = "rrf"        — Reciprocal Rank Fusion. Ignores scores entirely;
                        uses RANK only.
                        final = Σ 1 / (rrf_k + rank_i(c))
                        No normalisation required. Robust to score-scale drift.

The most common bug in hybrid implementations is forgetting that dense scores
live in [0, 1] but BM25 scores can be 0 to 10+ unbounded. Adding them raw
means BM25 always wins. We normalise per query before fusing in "weighted"
mode — RRF doesn't need normalisation because it never looks at the score.

Production patterns reinforced (S23):
  - SCORE NORMALISATION BEFORE FUSION (named pattern)
  - retriever strategy pattern — wraps two retrievers behind one pick()
  - pipeline logging with [hybrid] prefix
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import Retriever, Retrieved
from .similarity import SimilarityRetriever
from .bm25 import BM25Retriever, bm25_score_all, tokenize


# ── Hybrid constants ────────────────────────────────────────────────────────
DEFAULT_ALPHA = 0.5     # 0 = pure BM25, 1 = pure dense, 0.5 = balanced.
DEFAULT_RRF_K = 60      # Standard published RRF constant. Higher = smoother.
DEFAULT_FETCH_K = 30    # How many candidates each retriever brings to fusion.


class HybridRetriever(Retriever):
    """Hybrid dense + sparse retriever.

    Mode is fixed per instance. Switch by constructing a new one — the cost
    is microseconds because both inner retrievers share the same ChunkIndex.
    """

    name = "hybrid"

    def __init__(self, index, mode: str = "weighted",
                 alpha: float = DEFAULT_ALPHA,
                 rrf_k: int = DEFAULT_RRF_K,
                 fetch_k: int = DEFAULT_FETCH_K) -> None:
        if mode not in ("weighted", "rrf"):
            raise ValueError(f"mode must be 'weighted' or 'rrf', got {mode!r}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        super().__init__(index)
        self.mode = mode
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k

        # Reuse the existing single-signal retrievers. Same ChunkIndex.
        self._dense = SimilarityRetriever(index)
        self._sparse = BM25Retriever(index)

    def pick(self, query: str, k: int = 3, **kwargs: Any) -> List[Retrieved]:
        if len(self.index) == 0:
            return []

        if self.mode == "weighted":
            return self._pick_weighted(query, k)
        else:
            return self._pick_rrf(query, k)

    # ── Mode 1 — weighted α-fusion (with score normalisation) ───────────────

    def _pick_weighted(self, query: str, k: int) -> List[Retrieved]:
        """Min-max normalise both score vectors, then α-blend, then top-K."""
        # Get the FULL score array from each retriever. Not just top-K —
        # the fusion needs every chunk on the same axis.
        q_vec = self._dense._embed_query(query)
        dense_raw = self._dense._cosine_scores(q_vec)
        sparse_raw = bm25_score_all(self._sparse, query)

        dense_norm = _minmax(dense_raw)
        sparse_norm = _minmax(sparse_raw)

        fused = self.alpha * dense_norm + (1.0 - self.alpha) * sparse_norm

        top = np.argsort(-fused)[:k]
        results = [Retrieved(chunk=self.index.chunks[i],
                             score=float(fused[i]))
                   for i in top if fused[i] > 0]

        print(f"[hybrid:weighted]  α={self.alpha}  query={query!r}  "
              f"top-{k} fused scores: "
              + ", ".join(f"{r.score:.3f}" for r in results))
        return results

    # ── Mode 2 — Reciprocal Rank Fusion ─────────────────────────────────────

    def _pick_rrf(self, query: str, k: int) -> List[Retrieved]:
        """Rank-based fusion. RRF needs RANKS, not raw scores."""
        # Pull a candidate pool from each retriever — RRF is cheap, but
        # there's no point ranking chunks neither retriever liked.
        dense_hits = self._dense.pick(query, k=self.fetch_k)
        sparse_hits = self._sparse.pick(query, k=self.fetch_k)

        # rrf_scores[chunk_id] = Σ over retrievers of 1 / (rrf_k + rank)
        rrf_scores: Dict[int, float] = {}
        for hits in (dense_hits, sparse_hits):
            for rank, hit in enumerate(hits, start=1):
                cid = hit.chunk.chunk_id
                rrf_scores[cid] = (rrf_scores.get(cid, 0.0)
                                   + 1.0 / (self.rrf_k + rank))

        if not rrf_scores:
            print(f"[hybrid:rrf] WARN: no candidates from either retriever")
            return []

        # Map chunk_id → Chunk object (the ChunkIndex is the source of truth).
        id_to_chunk = {c.chunk_id: c for c in self.index.chunks}
        ordered = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
        results = [Retrieved(chunk=id_to_chunk[cid], score=float(score))
                   for cid, score in ordered if cid in id_to_chunk]

        print(f"[hybrid:rrf]  rrf_k={self.rrf_k}  query={query!r}  "
              f"top-{k} rrf scores: "
              + ", ".join(f"{r.score:.4f}" for r in results))
        return results


# ── Helpers ─────────────────────────────────────────────────────────────────

def _minmax(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1].  Per-query, not global.

    This is THE pattern that makes weighted hybrid work. Without it, BM25
    scores (0 to 10+) will dominate dense scores (0 to 1) and α stops
    mattering.
    """
    if scores.size == 0:
        return scores
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-12:
        # All scores identical — return zeros. This term contributes nothing.
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


# Optional CLI smoke test:  python -m retrievers.hybrid "your query"
if __name__ == "__main__":
    import os
    import sys
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, HERE)
    from loaders import load_corpus
    from chunker import chunk_corpus
    from retrievers.base import ChunkIndex

    files = [
        os.path.join(HERE, "data", "intro_to_rag.pdf"),
        os.path.join(HERE, "data", "product_manual.pdf"),
        os.path.join(HERE, "data", "sample_article.html"),
    ]
    docs = load_corpus(files)
    chunks = chunk_corpus(docs)
    idx = ChunkIndex()
    idx.add(chunks)

    q = " ".join(sys.argv[1:]) or "install widgetmax 3000"
    print()
    print("─" * 60)
    print("WEIGHTED  (α = 0.5)")
    print("─" * 60)
    for hit in HybridRetriever(idx, mode="weighted", alpha=0.5).pick(q, k=3):
        print(" ", hit)
    print()
    print("─" * 60)
    print("RRF")
    print("─" * 60)
    for hit in HybridRetriever(idx, mode="rrf").pick(q, k=3):
        print(" ", hit)
