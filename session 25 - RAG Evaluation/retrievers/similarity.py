"""
Session 22 — retrievers/similarity.py

STRATEGY 1 — pure similarity.

The S20 baseline, ported to the new retriever strategy interface. Embed the
query, score every chunk, return the top-K. No filtering, no diversity.

This is the right answer surprisingly often:
  - small clean corpus
  - one source type
  - a query whose answer fits in a single chunk

It's the wrong answer when:
  - the corpus has multiple source types and the question implies one
  - the top-K is dominated by near-duplicates that all repeat the same fact
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from .base import Retriever, Retrieved


class SimilarityRetriever(Retriever):
    """Pure top-K cosine similarity. The S20 retriever, named as a strategy."""

    name = "similarity"

    def pick(self, query: str, k: int = 3, **kwargs: Any) -> List[Retrieved]:
        if len(self.index) == 0:
            return []

        q_vec = self._embed_query(query)
        scores = self._cosine_scores(q_vec)

        top = np.argsort(-scores)[:k]
        results = [Retrieved(chunk=self.index.chunks[i],
                             score=float(scores[i]))
                   for i in top]

        print(f"[similarity] query={query!r}  top-{k} scores: "
              + ", ".join(f"{r.score:.3f}" for r in results))
        return results


# Optional CLI smoke test:  python -m retrievers.similarity "your query"
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

    q = " ".join(sys.argv[1:]) or "What is RAG?"
    r = SimilarityRetriever(idx)
    for hit in r.pick(q, k=3):
        print(" ", hit)
