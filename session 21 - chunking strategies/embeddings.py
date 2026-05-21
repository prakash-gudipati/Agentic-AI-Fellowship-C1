"""
Session 20 — embeddings.py

STAGE 3 of the RAG pipeline: EMBED.

We turn each chunk into a dense vector using OpenAI's text-embedding-3-small.
This is the same embedding model students saw in S18 + S19.

Production pattern reinforced:
  - same model used for ingest AND query (the rule from S19, never broken)
  - try/except on every external API call
  - env vars for secrets — never hardcode
  - batched calls are cheaper, so we batch by default
"""

import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ── Configuration ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"   # 1,536 dimensions, $0.02 / 1M tokens
EMBEDDING_BATCH = 64                         # how many strings per API call

load_dotenv()
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazy-init so importing the module does not require an API key."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings. Returns a (N, 1536) numpy array.

    Batches the requests to stay under the API per-request limit and
    to keep cost predictable.
    """
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)

    client = _get_client()
    all_vectors: List[List[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH):
        batch = texts[i:i + EMBEDDING_BATCH]
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        except Exception as e:
            # In production you would retry with backoff. S29 covers retry logic.
            raise RuntimeError(f"OpenAI embedding call failed: {e}") from e
        all_vectors.extend(item.embedding for item in resp.data)
        print(f"[embeddings]  embedded batch {i // EMBEDDING_BATCH + 1} "
              f"({len(batch)} texts)  model={EMBEDDING_MODEL}")

    matrix = np.asarray(all_vectors, dtype=np.float32)
    print(f"[embeddings]  total: {matrix.shape[0]} vectors, dim={matrix.shape[1]}")
    return matrix


def embed_one(text: str) -> np.ndarray:
    """Embed a single string. Returns a (1536,) numpy array. Used for queries."""
    return embed_texts([text])[0]


if __name__ == "__main__":
    # Quick smoke test — run python embeddings.py (requires OPENAI_API_KEY)
    sample = [
        "What is RAG?",
        "How do I install the WidgetMax 3000?",
        "Why does fine-tuning fail at out-of-corpus questions?",
    ]
    vecs = embed_texts(sample)
    print("[embeddings]  shape:", vecs.shape)
    print("[embeddings]  first 5 dims of vector[0]:", vecs[0, :5])
