"""
Session 24 - patterns/contextual.py

PATTERN 2 - Contextual retrieval (Anthropic).

The problem: when we chunk a document, each chunk loses its surrounding
context. A chunk that says "the rate dropped to 12%" is meaningless to a
cosine retriever - it doesn't know WHAT rate, in WHICH document, in WHICH
quarter. Embeddings of orphaned chunks miss semantic meaning that lives
in the surrounding pages.

The fix: at INGEST time, ask an LLM to write a short contextual prefix for
each chunk based on the WHOLE document, then prepend it to the chunk text
BEFORE embedding. Anthropic published this approach in 2024 and reported
roughly a 35% reduction in retrieval failure rate.

Example:

  Original chunk:
    "The rate dropped to 12%, the lowest in three years."

  Contextual prefix (written by Claude):
    "This chunk is from the Q3 2024 earnings call transcript for Acme Corp,
     section 'Customer Retention Metrics'."

  Embedded text:
    "This chunk is from the Q3 2024 earnings call transcript for Acme Corp,
     section 'Customer Retention Metrics'.

     The rate dropped to 12%, the lowest in three years."

The query "what was Acme's customer churn in Q3 2024" now matches.

Cost model:
  - One LLM call per chunk at INGEST. For a 100,000-chunk corpus, that's
    100k calls. Sounds expensive - but with PROMPT CACHING (S16), the
    document body is reused across all chunks in that document, so the
    per-chunk marginal cost collapses to about $0.0002.
  - At QUERY time: zero added cost. The embeddings are pre-baked.

This is the canonical INDEX-TIME INVESTMENT FOR QUERY-TIME GAINS pattern.

Production patterns reinforced (S24):
  - INDEX-TIME INVESTMENT FOR QUERY-TIME GAINS (named pattern, NEW today)
  - prompt caching (S16 callback)
  - try/except on every external call
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from dotenv import load_dotenv

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from chunker import Chunk
from loaders import Doc


load_dotenv()


# ----- Defaults ---------------------------------------------------------------
CONTEXT_MODEL = "claude-haiku-4-5-20251001"
CONTEXT_MAX_TOKENS = 120

CONTEXT_PROMPT = (
    "<document>\n{document}\n</document>\n\n"
    "Here is one chunk extracted from the document above:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Write a short (1-2 sentence) contextual prefix that situates this chunk "
    "inside the document - what the document is, what section the chunk is "
    "from, and what the chunk is about. Return ONLY the prefix, nothing else."
)


# ----- The contextualiser -----------------------------------------------------

def build_context_for_chunk(document: str, chunk: str,
                            model: str = CONTEXT_MODEL,
                            client=None) -> str:
    """Call the LLM to produce a 1-2 sentence contextual prefix.

    In production you would batch these calls with prompt caching - the
    document body is identical across every chunk in that document. We
    keep the demo readable by issuing one call per chunk.
    """
    if client is None:
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
        client = Anthropic(api_key=api_key)

    prompt = CONTEXT_PROMPT.format(document=document, chunk=chunk)
    resp = client.messages.create(
        model=model,
        max_tokens=CONTEXT_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(blk.text for blk in resp.content
                   if getattr(blk, "type", None) == "text")
    return text.strip()


# ----- The chunker wrapper ----------------------------------------------------

class ContextualChunker:
    """Wraps any base chunker. At chunking time, also writes a contextual
    prefix for each chunk and stores the combined string as the chunk text.

    The base_chunker_fn must follow the signature:
        base_chunker_fn(docs: List[Doc]) -> List[Chunk]

    This is identical to the chunker strategy pattern from S21/S23.
    """

    def __init__(self, base_chunker_fn,
                 model: str = CONTEXT_MODEL,
                 client=None,
                 prefix_separator: str = "\n\n") -> None:
        self.base_chunker_fn = base_chunker_fn
        self.model = model
        self.client = client
        self.prefix_separator = prefix_separator

    def chunk(self, docs: List[Doc]) -> List[Chunk]:
        """Chunk every doc, then attach a context prefix to each chunk."""
        chunks = self.base_chunker_fn(docs)
        # Build a docs lookup so we can find the parent doc per chunk.
        doc_by_source = {d.source: d for d in docs}

        print(f"[contextual] writing prefixes for {len(chunks)} chunks  "
              f"(model={self.model}) - this issues one LLM call per chunk")

        out: List[Chunk] = []
        for i, c in enumerate(chunks):
            doc = doc_by_source.get(c.source)
            if doc is None:
                # No matching doc - just keep the chunk as-is.
                out.append(c)
                continue
            try:
                prefix = build_context_for_chunk(
                    document=doc.text,
                    chunk=c.text,
                    model=self.model,
                    client=self.client,
                )
            except Exception as e:
                print(f"[contextual] WARN: chunk {i} failed ({e!s}) - "
                      f"keeping original chunk text")
                out.append(c)
                continue

            combined = prefix + self.prefix_separator + c.text
            out.append(Chunk(
                chunk_id=c.chunk_id, source=c.source,
                text=combined,
                metadata={**c.metadata, "contextual_prefix": prefix},
            ))
            if (i + 1) % 10 == 0:
                print(f"[contextual] prefixed {i + 1}/{len(chunks)} chunks")
        print(f"[contextual] done - {len(out)} contextualised chunks")
        return out


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, HERE)
    from loaders import load_corpus
    from chunker import chunk_corpus

    files = [
        os.path.join(HERE, "data", "intro_to_rag.pdf"),
        os.path.join(HERE, "data", "product_manual.pdf"),
    ]
    docs = load_corpus(files)
    wrapped = ContextualChunker(chunk_corpus)
    out = wrapped.chunk(docs)
    print()
    for c in out[:3]:
        print("---")
        print(c.text[:300])
        print()
