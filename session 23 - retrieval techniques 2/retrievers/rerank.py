"""
Session 23 — retrievers/rerank.py

STRATEGY 6 — RERANKING. The two-stage pattern.

Every retriever before today did a SINGLE pass — one scoring rule over
every chunk, top-K out. Reranking is the first multi-stage retriever:

  STAGE 1   first-pass retriever     →   N candidates  (N ≫ K)
  STAGE 2   reranker scores each     →   reordered list, top-K returned

WHY two stages?
  Calling an LLM on 50,000 chunks per query is impossible. Calling cosine
  on 50,000 chunks is cheap. The first stage exists to make the second
  stage AFFORDABLE. Recall first (wide cheap net), precision second
  (narrow expensive net).

This implementation uses Claude as the reranker — the "LLM as judge"
pattern from S14, now applied inside retrieval. We send the query plus
each candidate chunk, ask for a 0–10 relevance score, then reorder.

In a production system you would more often use a CROSS-ENCODER reranker
(BGE-reranker, Cohere rerank, jina-reranker) — purpose-built models that
are much cheaper than an LLM and slightly better at this specific task.
The IDEA is identical; only the scoring engine changes.

Production patterns reinforced (S23):
  - TWO-STAGE RETRIEVAL (named pattern)
  - LLM-AS-JUDGE  (called back from S14)
  - try/except on every external call
  - retrieve cheap-and-wide, score expensive-and-narrow
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from dotenv import load_dotenv

from .base import Retriever, Retrieved


# ── Configuration ───────────────────────────────────────────────────────────
DEFAULT_FETCH_K = 10       # candidates from the first-pass retriever
DEFAULT_MAX_TOKENS = 800   # tight cap — judges return short JSON
RERANKER_MODEL = "claude-haiku-4-5-20251001"   # cheap, fast, plenty for judging

load_dotenv()


# ── Reranker prompt ─────────────────────────────────────────────────────────
# Tight, structured, returns a single score per candidate.
JUDGE_SYSTEM = (
    "You are a retrieval judge. You are given a user query and N candidate "
    "passages from a document corpus. Score each passage from 0 to 10 by how "
    "well it answers the user's query — 10 = direct answer, 0 = unrelated. "
    "Return ONLY a JSON array of N integers. No prose. No keys. No markdown."
)

JUDGE_USER_TEMPLATE = (
    "QUERY:\n{query}\n\n"
    "CANDIDATES (numbered):\n{candidates}\n\n"
    "Respond with a JSON array of {n} integers between 0 and 10, in the same "
    "order as the candidates."
)


class LLMRerankRetriever(Retriever):
    """Two-stage retriever — base retriever + LLM reranker.

    Wraps any Retriever (similarity, bm25, hybrid — anything that conforms
    to the strategy interface). At pick() time:
      1. Ask the base retriever for `fetch_k` candidates.
      2. Send them to Claude as a single prompt; parse N integer scores.
      3. Reorder by judge score; return the top-K.

    On any failure of stage 2 we fall back to the stage-1 order. That keeps
    the pipeline alive when the model is down — production discipline.
    """

    name = "rerank"

    def __init__(self, index, base: Optional[Retriever] = None,
                 fetch_k: int = DEFAULT_FETCH_K,
                 model: str = RERANKER_MODEL) -> None:
        super().__init__(index)
        # Default base is plain cosine — cheap, broad recall.
        from .similarity import SimilarityRetriever
        self.base = base if base is not None else SimilarityRetriever(index)
        self.fetch_k = fetch_k
        self.model = model
        self._client = None  # lazy

    def _get_client(self):
        """Lazy Anthropic client — module imports must not require an API key."""
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError("anthropic package not installed. "
                               "pip install anthropic") from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env "
                "and add your key."
            )
        self._client = Anthropic(api_key=api_key)
        return self._client

    def pick(self, query: str, k: int = 3, **kwargs: Any) -> List[Retrieved]:
        # ── STAGE 1 ─ first-pass recall ─────────────────────────────────────
        candidates = self.base.pick(query, k=self.fetch_k)
        if not candidates:
            print(f"[rerank] base retriever returned 0 candidates  "
                  f"query={query!r}")
            return []
        if len(candidates) <= k:
            print(f"[rerank] only {len(candidates)} candidates ≤ k={k}, "
                  f"skipping stage 2")
            return candidates

        # ── STAGE 2 ─ LLM reranks the candidates ────────────────────────────
        try:
            scores = self._llm_score(query, candidates)
        except Exception as e:
            print(f"[rerank] WARN: stage 2 failed ({e!s}) — falling back "
                  f"to stage 1 order")
            return candidates[:k]

        # Pair every candidate with its judge score and sort descending.
        # On length mismatch (bad JSON), pad/truncate with original ranks.
        if len(scores) != len(candidates):
            print(f"[rerank] WARN: judge returned {len(scores)} scores for "
                  f"{len(candidates)} candidates — falling back to stage 1")
            return candidates[:k]

        scored: List[Retrieved] = [
            Retrieved(chunk=cand.chunk, score=float(scores[i]))
            for i, cand in enumerate(candidates)
        ]
        scored.sort(key=lambda r: -r.score)
        out = scored[:k]

        print(f"[rerank] query={query!r}  model={self.model}  "
              f"fetch_k={self.fetch_k}  top-{k} judge scores: "
              + ", ".join(f"{r.score:.1f}" for r in out))
        return out

    # ── LLM call ────────────────────────────────────────────────────────────

    def _llm_score(self, query: str, candidates: List[Retrieved]) -> List[int]:
        """Send query + candidates to the model. Parse a JSON int array."""
        client = self._get_client()
        numbered = "\n\n".join(
            f"[{i + 1}] {self._snippet(c.chunk.text)}"
            for i, c in enumerate(candidates)
        )
        prompt = JUDGE_USER_TEMPLATE.format(
            query=query, candidates=numbered, n=len(candidates)
        )

        resp = client.messages.create(
            model=self.model,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(blk.text for blk in resp.content
                       if getattr(blk, "type", None) == "text")
        return _parse_score_array(text)

    @staticmethod
    def _snippet(text: str, max_chars: int = 600) -> str:
        """Trim very long chunks so the judge prompt stays compact."""
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]


# ── JSON extraction ─────────────────────────────────────────────────────────

_JSON_ARRAY_RE = re.compile(r"\[\s*(?:-?\d+\s*,\s*)*-?\d+\s*\]")


def _parse_score_array(text: str) -> List[int]:
    """Extract the first JSON int array from `text`. Clamp every score to 0..10.

    The judge prompt asks for "ONLY a JSON array", but real models sometimes
    add a stray sentence. We isolate the array with a regex first.
    """
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        raise ValueError(f"no JSON int array found in judge output: {text!r}")
    arr = json.loads(match.group(0))
    if not isinstance(arr, list):
        raise ValueError("judge output is not a JSON list")
    out: List[int] = []
    for v in arr:
        try:
            iv = int(v)
        except Exception:
            raise ValueError(f"judge score is not an int: {v!r}")
        out.append(max(0, min(10, iv)))
    return out


# Optional CLI smoke test:  python -m retrievers.rerank "your query"
if __name__ == "__main__":
    import sys
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, HERE)
    from loaders import load_corpus
    from chunker import chunk_corpus
    from retrievers.base import ChunkIndex
    from retrievers.similarity import SimilarityRetriever

    files = [
        os.path.join(HERE, "data", "intro_to_rag.pdf"),
        os.path.join(HERE, "data", "product_manual.pdf"),
        os.path.join(HERE, "data", "sample_article.html"),
    ]
    docs = load_corpus(files)
    chunks = chunk_corpus(docs)
    idx = ChunkIndex()
    idx.add(chunks)

    base = SimilarityRetriever(idx)
    rer = LLMRerankRetriever(idx, base=base, fetch_k=10)

    q = " ".join(sys.argv[1:]) or "What is RAG?"
    print()
    print("── STAGE 1 (similarity, fetch_k=10) ──")
    for hit in base.pick(q, k=10):
        print(" ", hit)
    print()
    print("── STAGE 2 (LLM rerank → top 3) ──")
    for hit in rer.pick(q, k=3):
        print(" ", hit)
