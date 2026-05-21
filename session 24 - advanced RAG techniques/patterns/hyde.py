"""
Session 24 - patterns/hyde.py

PATTERN 3 - HyDE (Hypothetical Document Embeddings).

The query-document gap: queries ("how do I install the WidgetMax?") and
documents ("Step 1. Press the power button for three seconds. The front
LED will turn solid green when...") live in different language registers.
Cosine sometimes can't bridge them - the query asks; the document answers.

HyDE's trick: instead of embedding the QUERY and searching for matching
documents, embed a FAKE ANSWER and search with that.

  1. User types a query.
  2. We send it to an LLM with a prompt like:
       "Write a short passage that would answer this query as if it appeared
        in a real document."
  3. The LLM produces a hypothetical document - a fake answer.
  4. We embed the HYPOTHETICAL DOCUMENT, not the query.
  5. We search with that embedding.

Why it works: the hypothetical doc lives in answer-space, the same space
real documents occupy. The query lives in question-space, a different
register. Vectors of answers cluster with vectors of answers.

When it backfires: when the query is ambiguous, the LLM hallucinates a
plausible-but-wrong direction, and HyDE confidently retrieves chunks that
match the WRONG topic. Always test on your corpus before shipping.

Production patterns reinforced (S24):
  - retriever strategy pattern (S22) - HyDERetriever wraps any base retriever
  - QUERY REWRITING BEFORE RETRIEVAL (named pattern, NEW today)
  - try/except on every external call
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional

from dotenv import load_dotenv

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from retrievers.base import Retriever, Retrieved
from retrievers.similarity import SimilarityRetriever


load_dotenv()


# ----- Defaults ---------------------------------------------------------------
HYDE_MODEL = "claude-haiku-4-5-20251001"
HYDE_MAX_TOKENS = 300

HYDE_PROMPT = (
    "Write a short paragraph (3-5 sentences) that would answer the user's "
    "query, written as if it were directly extracted from a relevant "
    "document or manual. Do NOT mention that you are hypothesising. Write "
    "in the third person, in a technical encyclopedic style.\n\n"
    "USER QUERY:\n{query}\n\n"
    "HYPOTHETICAL DOCUMENT PASSAGE:"
)


class HyDERetriever(Retriever):
    """Wrap any base retriever. At pick() time:

      1. Ask the LLM for a hypothetical document passage answering the query.
      2. Embed the hypothetical passage instead of the query.
      3. Run the base retriever using the hypothetical passage as the
         'query' string (since the base retriever embeds its query input).
      4. Return the base retriever's results.

    HyDERetriever does NOT need its own embedding code - it delegates to
    the base retriever, which uses the same embed_one() from embeddings.py.
    """

    name = "hyde"

    def __init__(self, index, base: Optional[Retriever] = None,
                 model: str = HYDE_MODEL) -> None:
        super().__init__(index)
        self.base = base if base is not None else SimilarityRetriever(index)
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

        try:
            hypothetical = self._generate_hypothetical(query)
        except Exception as e:
            print(f"[hyde] WARN: hypothetical-doc generation failed "
                  f"({e!s}) - falling back to plain base retrieval")
            return self.base.pick(query, k=k)

        print(f"[hyde] query={query!r}")
        print(f"[hyde] hypothetical: {hypothetical[:120]!r}...")

        # Hand the hypothetical doc to the base retriever AS the query.
        # Since the base retriever embeds its query input via embed_one(),
        # this effectively embeds the hypothetical doc instead.
        return self.base.pick(hypothetical, k=k)

    def _generate_hypothetical(self, query: str) -> str:
        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=HYDE_MAX_TOKENS,
            messages=[{"role": "user",
                       "content": HYDE_PROMPT.format(query=query)}],
        )
        text = "".join(blk.text for blk in resp.content
                       if getattr(blk, "type", None) == "text")
        return text.strip()


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
    r = HyDERetriever(idx, base=base)

    q = " ".join(sys.argv[1:]) or "When should I use retrieval over fine-tuning?"
    print()
    print("---- basic similarity ----")
    for hit in base.pick(q, k=3):
        print(f"  {hit.chunk.source}  score={hit.score:.3f}")
    print()
    print("---- HyDE ----")
    for hit in r.pick(q, k=3):
        print(f"  {hit.chunk.source}  score={hit.score:.3f}")
