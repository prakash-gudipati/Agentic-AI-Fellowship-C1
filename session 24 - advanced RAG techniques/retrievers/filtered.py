"""
Session 22 — retrievers/filtered.py

STRATEGY 2 — metadata filtering, then similarity.

The pattern: PRE-FILTER the candidate pool with a cheap rule, then SCORE
what's left with the expensive rule. This is universal in production search
systems. Bouncer at the door, not surveying everyone inside.

Today's filter is `equals` on chunk metadata fields, e.g.

    retriever.pick(q, k=3, filter={"source_type": "manual"})

is read as "only chunks whose metadata['source_type'] equals 'manual'".

Multiple keys in the filter dict are combined with AND. Values can be either
a single value (equals match) or a list (any-of match).

When this strategy wins:
  - corpus has multiple source types and the user implicitly chose one
    ("how do I install" → manual; "explain the theory" → paper)
  - corpus has dates, authors, languages — anything that narrows fast
  - similarity alone returns chunks from the wrong sub-corpus

Edge case: the filter is too strict and matches zero chunks. We surface that
with an empty result + a clear log line, rather than fall back silently.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .base import Retriever, Retrieved


class FilteredRetriever(Retriever):
    """Metadata pre-filter, then top-K cosine."""

    name = "filtered"

    def pick(self, query: str, k: int = 3,
             filter: Optional[Dict[str, Any]] = None,
             **kwargs: Any) -> List[Retrieved]:
        if len(self.index) == 0:
            return []

        # Step 1 — pre-filter. Build a boolean mask over the chunk array.
        mask = self._build_mask(filter)
        kept = int(mask.sum())
        total = len(self.index)
        print(f"[filtered] filter={filter}  "
              f"candidates: {kept}/{total} chunks pass")

        if kept == 0:
            print("[filtered] WARN: zero chunks pass the filter — "
                  "returning empty result")
            return []

        # Step 2 — score only the survivors. Cheaper than scoring all chunks
        # for very narrow filters, identical when the filter is empty.
        q_vec = self._embed_query(query)
        scores = self._cosine_scores(q_vec)
        scores = np.where(mask, scores, -np.inf)  # block out non-matches

        top = np.argsort(-scores)[:k]
        # Drop any entries that ended up at -inf (when fewer than k pass).
        results = []
        for i in top:
            if not np.isfinite(scores[i]):
                break
            results.append(Retrieved(chunk=self.index.chunks[i],
                                     score=float(scores[i])))

        print(f"[filtered] query={query!r}  top-{k} scores: "
              + ", ".join(f"{r.score:.3f}" for r in results))
        return results

    # ── Filter helpers ──────────────────────────────────────────────────────

    def _build_mask(self, filter: Optional[Dict[str, Any]]) -> np.ndarray:
        """Boolean mask of chunks that satisfy every key in `filter`."""
        n = len(self.index)
        mask = np.ones(n, dtype=bool)
        if not filter:
            return mask

        for key, expected in filter.items():
            row = np.zeros(n, dtype=bool)
            allowed = _as_set(expected)
            for i, chunk in enumerate(self.index.chunks):
                actual = chunk.metadata.get(key)
                if actual in allowed:
                    row[i] = True
            mask &= row
        return mask


def _as_set(value: Any) -> set:
    """Accept either a single value or any iterable of values."""
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

    r = FilteredRetriever(idx)
    print()
    print("Filter: source_type=manual")
    for hit in r.pick("How do I install it?", k=3,
                      filter={"source_type": "manual"}):
        print(" ", hit)
