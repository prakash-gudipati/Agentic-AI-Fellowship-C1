"""
Session 22 / 27 -- retrievers/mmr.py

STRATEGY 2 -- Maximal Marginal Relevance (MMR).

WHY THIS RETRIEVER EXISTS
-------------------------
Pure similarity returns the top-K by relevance to the query. That's
the right answer when the corpus is diverse. It's the WRONG answer
when the top-K is dominated by near-duplicates that all repeat the
same fact -- because then your LLM sees the same evidence k times
instead of k different pieces of evidence.

MMR re-ranks the candidate pool to balance:
  - relevance to the query
  - dissimilarity to already-selected results

The lambda knob controls the trade-off:
  lambda = 1.0  -> pure similarity (no diversity bonus)
  lambda = 0.0  -> pure diversity (relevance ignored)
  lambda = 0.6  -> the production sweet spot for most corpora

ALGORITHM
---------
1. Pull a candidate pool ("fetch_k") via pure similarity.
2. Greedily select k results, scoring each candidate at each step as:
       score = lambda * sim(query, candidate)
             - (1 - lambda) * max( sim(candidate, already_selected) )
3. The first pick is the highest-similarity candidate. Each subsequent
   pick balances "relevant to the query" against "different from what
   we already have."
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from .base import Retriever, Retrieved


class MMRRetriever(Retriever):
    """Maximal Marginal Relevance retriever.

    Args:
        fetch_k:  size of the candidate pool. Typically 3-5x k.
        lambda_:  relevance/diversity trade-off (1.0 = pure relevance,
                  0.0 = pure diversity).
    """

    name = "mmr"

    def __init__(self, index, *, fetch_k: int = 12,
                 lambda_: float = 0.6) -> None:
        super().__init__(index)
        self.fetch_k = fetch_k
        self.lambda_ = float(lambda_)

    def pick(self, query: str, k: int = 3,
             **kwargs: Any) -> List[Retrieved]:
        if len(self.index) == 0:
            return []

        q_vec = self._embed_query(query)
        sims_to_q = self._cosine_scores(q_vec)

        # Candidate pool: top-fetch_k by raw similarity.
        pool_idx = np.argsort(-sims_to_q)[: self.fetch_k].tolist()
        if not pool_idx:
            return []

        selected: List[int] = []
        remaining = list(pool_idx)
        lam = self.lambda_

        # Pre-compute pairwise similarity between candidates so the
        # inner loop is matrix lookup rather than re-dot.
        cand_vecs = self.index.vectors[pool_idx]
        pair_sims = cand_vecs @ cand_vecs.T

        # Map global chunk-index -> position inside the pool.
        pool_position = {idx: i for i, idx in enumerate(pool_idx)}

        for _ in range(min(k, len(remaining))):
            best_score = -1e9
            best_idx = remaining[0]
            for idx in remaining:
                rel = sims_to_q[idx]
                if selected:
                    sel_positions = [pool_position[s] for s in selected]
                    div_penalty = float(
                        pair_sims[pool_position[idx], sel_positions].max())
                else:
                    div_penalty = 0.0
                score = lam * rel - (1.0 - lam) * div_penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)

        results = [Retrieved(chunk=self.index.chunks[i],
                              score=float(sims_to_q[i]))
                    for i in selected]
        print(f"[mmr] query={query!r}  lambda={lam:.2f}  "
              f"top-{k} sims: "
              + ", ".join(f"{r.score:.3f}" for r in results))
        return results
