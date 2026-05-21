"""
Session 24 - patterns/multi_query.py

PATTERN 4 - Multi-query retrieval (query expansion).

The query-phrasing problem: cosine and BM25 are both sensitive to wording.
"How do I install the WidgetMax" and "WidgetMax installation instructions"
embed close but not identically, and may return different top-K. BM25 is
even more brittle - it cares about literal tokens.

Multi-query fix:
  1. Generate N paraphrases of the user query via the LLM.
  2. Run the base retriever once per variant.
  3. Fuse the result lists with RECIPROCAL RANK FUSION (S23 callback).
  4. Return the fused top-K.

Effectively the system votes - chunks that come up in multiple variants
rank higher than chunks that only come up in one variant.

We deliberately reuse RRF instead of weighted fusion because we have no
labelled queries to tune alpha against, and RRF is robust to score scales
across retrievers.

Production patterns reinforced (S24):
  - retriever strategy pattern (S22) - wraps any base retriever
  - QUERY REWRITING BEFORE RETRIEVAL (named pattern, NEW today)
  - RRF for fusion (S23 callback)
  - try/except on every external call
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from retrievers.base import Retriever, Retrieved
from retrievers.similarity import SimilarityRetriever


load_dotenv()


# ----- Defaults ---------------------------------------------------------------
EXPAND_MODEL = "claude-haiku-4-5-20251001"
EXPAND_MAX_TOKENS = 400
DEFAULT_VARIANTS = 4
DEFAULT_RRF_K = 60
DEFAULT_FETCH_K = 10

EXPAND_PROMPT = (
    "Generate {n} different phrasings of the user query below. The phrasings "
    "should preserve the meaning but vary in vocabulary, level of detail, "
    "and angle. Return ONLY a JSON array of {n} strings. No prose. No keys.\n\n"
    "USER QUERY: {query}"
)

_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", flags=re.DOTALL)


class MultiQueryRetriever(Retriever):
    """Wrap any base retriever. Generate N query variants, retrieve from each,
    fuse with RRF.

    `n_variants` is the only user-facing knob besides the base retriever.
    Higher n -> more recall, more cost. n = 4 is a good default.
    """

    name = "multi_query"

    def __init__(self, index, base: Optional[Retriever] = None,
                 n_variants: int = DEFAULT_VARIANTS,
                 rrf_k: int = DEFAULT_RRF_K,
                 fetch_k: int = DEFAULT_FETCH_K,
                 model: str = EXPAND_MODEL) -> None:
        super().__init__(index)
        self.base = base if base is not None else SimilarityRetriever(index)
        self.n_variants = n_variants
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k
        self.model = model
        self._client = None

    def _get_client(self):
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
        if len(self.index) == 0:
            return []

        # Generate variants. Always include the original query as one of
        # the inputs to retrieval - safest default.
        try:
            variants = self._generate_variants(query)
        except Exception as e:
            print(f"[multi_query] WARN: variant generation failed ({e!s}) - "
                  f"falling back to plain base retrieval")
            return self.base.pick(query, k=k)

        all_queries = [query] + variants
        print(f"[multi_query] generated {len(variants)} variants -> "
              f"running {len(all_queries)} retrievals")

        # RRF fuse across query results.
        rrf_scores: Dict[int, float] = {}
        seen_chunks: Dict[int, Any] = {}
        for q in all_queries:
            hits = self.base.pick(q, k=self.fetch_k)
            for rank, hit in enumerate(hits, start=1):
                cid = hit.chunk.chunk_id
                seen_chunks[cid] = hit.chunk
                rrf_scores[cid] = (rrf_scores.get(cid, 0.0)
                                   + 1.0 / (self.rrf_k + rank))

        if not rrf_scores:
            return []

        ordered = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
        out = [Retrieved(chunk=seen_chunks[cid], score=float(score))
               for cid, score in ordered]
        print(f"[multi_query] rrf top-{k}: "
              + ", ".join(f"{r.score:.4f}" for r in out))
        return out

    # ----- LLM call ----------------------------------------------------------

    def _generate_variants(self, query: str) -> List[str]:
        client = self._get_client()
        prompt = EXPAND_PROMPT.format(n=self.n_variants, query=query)
        resp = client.messages.create(
            model=self.model,
            max_tokens=EXPAND_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(blk.text for blk in resp.content
                       if getattr(blk, "type", None) == "text")
        return _parse_string_array(text)


def _parse_string_array(text: str) -> List[str]:
    """Extract the first JSON string array. Robust to surrounding prose."""
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        raise ValueError(f"no JSON array found in variant output: {text!r}")
    arr = json.loads(match.group(0))
    if not isinstance(arr, list):
        raise ValueError("variant output is not a list")
    return [str(v).strip() for v in arr if str(v).strip()]


if __name__ == "__main__":
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
    idx = ChunkIndex(); idx.add(chunks)

    base = SimilarityRetriever(idx)
    r = MultiQueryRetriever(idx, base=base, n_variants=3)
    q = " ".join(sys.argv[1:]) or "How do I install the WidgetMax 3000?"
    print()
    for hit in r.pick(q, k=3):
        preview = hit.chunk.text[:80].replace("\n", " ")
        print(f"  score={hit.score:.4f}  src={hit.chunk.source}  {preview}")
