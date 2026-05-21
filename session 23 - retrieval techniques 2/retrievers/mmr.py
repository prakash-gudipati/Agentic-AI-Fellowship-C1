"""
Session 22 — retrievers/mmr.py

STRATEGY 3 — Maximal Marginal Relevance (MMR).

Pure similarity has a famous failure mode: the top-K is often three almost
identical chunks. The LLM gets nothing new from chunks 2 and 3 — they just
restate chunk 1 in different words.

MMR fixes this by SUBTRACTING redundancy from relevance. After picking the
single most-similar chunk, every subsequent pick scores candidates by:

    MMR(c) = λ · sim(c, query) - (1 - λ) · max_{p in picked} sim(c, p)
              ↑ relevance term       ↑ redundancy penalty

λ ∈ [0, 1] is the only knob.
  λ = 1.0  →  pure similarity (identical to SimilarityRetriever)
  λ = 0.0  →  pure diversity (picks the most-different chunks regardless of relevance)
  λ = 0.5  →  balanced (a good default for most corpora)

The algorithm is greedy — pick one, recompute, pick the next. K passes total.
Cost is O(K × N), where N is the candidate pool. Cheap.

This file accepts an optional `filter` so MMR composes with metadata filtering.
That single-line composition is the reason both retrievers share the same
ChunkIndex and base class.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import Retriever, Retrieved


# Default λ — balanced. Override per-call via pick(..., lambda_=0.7).
DEFAULT_LAMBDA = 0.5
# How many top-similarity chunks MMR considers as candidates. Bigger pool =
# more chance of finding diverse high-quality picks. Cost scales linearly.
DEFAULT_FETCH_K = 20


class MMRRetriever(Retriever):
    """Greedy Maximal Marginal Relevance retriever."""

    name = "mmr"

    def pick(self, query: str, k: int = 3,
             lambda_: float = DEFAULT_LAMBDA,
             fetch_k: int = DEFAULT_FETCH_K,
             filter: Optional[Dict[str, Any]] = None,
             **kwargs: Any) -> List[Retrieved]:
        if len(self.index) == 0:
            return []
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")

        # Step 1 — score every chunk against the query.
        q_vec = self._embed_query(query)
        rel_scores = self._cosine_scores(q_vec)

        # Optional metadata pre-filter. Same idea as FilteredRetriever — keep
        # the strategy composable.
        mask = self._build_mask(filter)
        rel_scores = np.where(mask, rel_scores, -np.inf)

        # Step 2 — narrow to a candidate pool of the fetch_k most-relevant.
        order = np.argsort(-rel_scores)
        candidate_idx = [int(i) for i in order[:fetch_k]
                         if np.isfinite(rel_scores[i])]
        if not candidate_idx:
            print("[mmr] WARN: zero candidates after filter — empty result")
            return []

        # Step 3 — greedy MMR over the candidate pool.
        picked: List[int] = []
        picked_scores: List[float] = []
        candidates = list(candidate_idx)

        # First pick is always the most relevant — no redundancy yet.
        first = candidates.pop(0)
        picked.append(first)
        picked_scores.append(float(rel_scores[first]))

        while len(picked) < k and candidates:
            # For each remaining candidate, redundancy = max similarity to
            # any already-picked chunk. Cosine of L2-normalised vectors == dot.
            picked_matrix = self.index.vectors[picked]            # (P, D)
            cand_matrix = self.index.vectors[candidates]          # (C, D)
            redundancy = (cand_matrix @ picked_matrix.T).max(axis=1)  # (C,)

            cand_rel = rel_scores[candidates]                     # (C,)
            mmr_scores = lambda_ * cand_rel - (1.0 - lambda_) * redundancy

            best_local = int(np.argmax(mmr_scores))
            best_global = candidates[best_local]
            picked.append(best_global)
            picked_scores.append(float(mmr_scores[best_local]))
            candidates.pop(best_local)

        results = [Retrieved(chunk=self.index.chunks[i],
                             score=picked_scores[n])
                   for n, i in enumerate(picked)]

        print(f"[mmr] query={query!r}  λ={lambda_}  "
              f"top-{k} mmr-scores: "
              + ", ".join(f"{r.score:.3f}" for r in results))
        return results

    # Reuse the same filter logic as FilteredRetriever to keep behaviour
    # identical. (In a real codebase we'd factor this out into a helper module.)
    def _build_mask(self, filter: Optional[Dict[str, Any]]) -> np.ndarray:
        n = len(self.index)
        mask = np.ones(n, dtype=bool)
        if not filter:
            return mask
        for key, expected in filter.items():
            row = np.zeros(n, dtype=bool)
            allowed = _as_set(expected)
            for i, chunk in enumerate(self.index.chunks):
                if chunk.metadata.get(key) in allowed:
                    row[i] = True
            mask &= row
        return mask


def _as_set(value: Any) -> set:
    if isinstance(value, (list, tuple, set, frozenset)):
        return set(value)
    return {value}


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

    r = MMRRetriever(idx)
    print()
    for lam in (1.0, 0.5, 0.0):
        print(f"\n── λ = {lam} ──")
        for hit in r.pick("What is RAG?", k=3, lambda_=lam):
            print(" ", hit)
