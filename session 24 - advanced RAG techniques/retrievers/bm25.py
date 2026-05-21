"""
Session 23 — retrievers/bm25.py

STRATEGY 4 — BM25 keyword retrieval.

The dense retrievers from S22 see MEANING. They don't see EXACT WORDS.
A query like "WX-3000 error E47" gets averaged across 1,536 dimensions
and the rare product code disappears — the retriever returns chunks about
"errors" or "products" instead of the chunk that literally mentions
"WX-3000" and "E47".

BM25 is the keyword retriever. It counts how many times each query word
appears in each chunk, weights rare words more (IDF), and caps the count
so 50 mentions don't beat 5 (TF saturation). It's the production keyword
formula — every serious search engine has it.

Two refinements on top of naïve TF × IDF:
  1. TF SATURATION    — diminishing returns on term frequency.
                        formula:  tf * (k1 + 1) / (tf + k1 * len_norm)
  2. LENGTH NORMALISE — a long chunk shouldn't beat a short one just by
                        sheer mass. Adjust by avg_doc_length.

We use the standard Okapi-BM25 parameters:
  k1 = 1.5   — controls TF saturation curve. Higher = saturates slower.
  b  = 0.75  — strength of length normalisation. 0 = ignore length, 1 = full.

Pure Python. No external BM25 library. Students see the actual loop.

Production patterns reinforced:
  - retriever strategy pattern (S22)
  - pre-compute at index time (term lists, IDF), score cheap at query time
  - pipeline logging with [bm25] prefix
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from .base import Retriever, Retrieved


# ── BM25 constants — these are the standard Okapi values ────────────────────
BM25_K1 = 1.5    # TF saturation. Higher = TF curve flattens slower.
BM25_B  = 0.75   # Length normalisation strength. 0 = off, 1 = full.

# Words too common to carry any signal — pulled out before scoring.
# Production systems use a longer language-aware stopword list (NLTK / spaCy).
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "as",
    "and", "or", "but", "not", "no", "do", "does", "did", "have", "has",
    "had", "this", "that", "these", "those", "it", "its", "i", "you",
    "we", "they", "he", "she", "them", "us", "my", "your", "our", "their",
    "what", "which", "who", "how", "when", "where", "why", "if", "then",
    "so", "too", "very", "can", "will", "just", "than", "about",
})

# Simple word-character tokenizer. Production systems use a real tokenizer
# (e.g. sentence-piece) — this is enough for a teaching demo.
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-word, drop stopwords. Returns a token list."""
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25Retriever(Retriever):
    """BM25 keyword retriever — sparse cousin of SimilarityRetriever.

    Index-time work (done once in __init__):
      - tokenise every chunk
      - count term frequencies per chunk
      - compute corpus-wide IDF for every term
      - record chunk lengths and avg chunk length

    Query-time work:
      - tokenise the query
      - sum BM25(term, chunk) for every query term, for every chunk
      - top-K by total score
    """

    name = "bm25"

    def __init__(self, index, k1: float = BM25_K1, b: float = BM25_B) -> None:
        super().__init__(index)
        self.k1 = k1
        self.b = b

        # Pre-compute everything that doesn't depend on the query.
        self._chunk_tokens: List[List[str]] = [
            tokenize(c.text) for c in index.chunks
        ]
        self._chunk_tfs: List[Counter] = [
            Counter(toks) for toks in self._chunk_tokens
        ]
        self._chunk_lens: np.ndarray = np.asarray(
            [len(toks) for toks in self._chunk_tokens], dtype=np.float32
        )
        self._avg_len: float = (float(self._chunk_lens.mean())
                                if len(self._chunk_lens) > 0 else 0.0)
        self._idf: Dict[str, float] = self._compute_idf()

        print(f"[bm25] index built  N={len(index)}  "
              f"vocab={len(self._idf)}  "
              f"avg_chunk_tokens={self._avg_len:.1f}")

    def _compute_idf(self) -> Dict[str, float]:
        """IDF using the BM25-Okapi formulation.

        idf(t) = log(  (N - n_t + 0.5) / (n_t + 0.5)  +  1  )

        N   = total number of chunks
        n_t = number of chunks the term t appears in (at least once)

        The "+1" inside the log keeps idf >= 0 for very common terms.
        """
        N = len(self._chunk_tokens)
        doc_freq: Counter = Counter()
        for tfs in self._chunk_tfs:
            for term in tfs:
                doc_freq[term] += 1
        idf: Dict[str, float] = {}
        for term, n_t in doc_freq.items():
            idf[term] = math.log(((N - n_t + 0.5) / (n_t + 0.5)) + 1.0)
        return idf

    def pick(self, query: str, k: int = 3, **kwargs: Any) -> List[Retrieved]:
        if len(self.index) == 0:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            print(f"[bm25] WARN: query has no scorable tokens: {query!r}")
            return []

        scores = self._score_all(q_tokens)

        top = np.argsort(-scores)[:k]
        results = [Retrieved(chunk=self.index.chunks[i],
                             score=float(scores[i]))
                   for i in top if scores[i] > 0]

        print(f"[bm25] query={query!r}  tokens={q_tokens}  "
              f"top-{k} scores: "
              + ", ".join(f"{r.score:.3f}" for r in results))
        return results

    # ── Internal scoring helpers ────────────────────────────────────────────

    def _score_all(self, q_tokens: List[str]) -> np.ndarray:
        """Sum BM25(term, chunk) over every query term, for every chunk."""
        n = len(self.index)
        scores = np.zeros(n, dtype=np.float32)
        for term in q_tokens:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue  # term unseen in corpus — contributes nothing
            for i, tfs in enumerate(self._chunk_tfs):
                tf = tfs.get(term, 0)
                if tf == 0:
                    continue
                scores[i] += idf * self._tf_contribution(tf, self._chunk_lens[i])
        return scores

    def _tf_contribution(self, tf: int, doc_len: float) -> float:
        """Okapi TF term — TF saturation + length normalisation.

        numerator   = tf · (k1 + 1)
        denominator = tf + k1 · (1 − b + b · doc_len / avg_len)
        """
        len_norm = 1.0 - self.b + self.b * (doc_len / self._avg_len
                                            if self._avg_len > 0 else 0.0)
        return (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)


# Public helper — exposed so HybridRetriever can build a BM25 score array
# for fusion without re-instantiating a Retriever object.
def bm25_score_all(retriever: "BM25Retriever", query: str) -> np.ndarray:
    """Return the full BM25 score array for every chunk in the index."""
    return retriever._score_all(tokenize(query))


# Optional CLI smoke test:  python -m retrievers.bm25 "your query"
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

    r = BM25Retriever(idx)
    q = " ".join(sys.argv[1:]) or "install widgetmax product"
    print()
    for hit in r.pick(q, k=3):
        print(" ", hit)
