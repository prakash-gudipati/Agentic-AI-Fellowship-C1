"""
Session 22 — embeddings.py

Unchanged from S20. Same OpenAI text-embedding-3-small model used for both
ingest AND query — the rule from S19, never broken.

Production patterns reinforced:
  - same model for ingest AND query (S19/S20)
  - try/except on every external API call
  - env vars for secrets
  - batched calls keep cost predictable
"""

import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ── Configuration ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"   # 1,536 dims
EMBEDDING_BATCH = 64

load_dotenv()
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazy-init so importing the module does not require an API key."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Copy .env.example to .env and add "
                "your key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings. Returns a (N, 1536) numpy array."""
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)

    client = _get_client()
    all_vectors: List[List[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH):
        batch = texts[i:i + EMBEDDING_BATCH]
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding call failed: {e}") from e
        all_vectors.extend(item.embedding for item in resp.data)
        print(f"[embeddings] embedded batch {i // EMBEDDING_BATCH + 1} "
              f"({len(batch)} texts)  model={EMBEDDING_MODEL}")

    matrix = np.asarray(all_vectors, dtype=np.float32)
    print(f"[embeddings] total: {matrix.shape[0]} vectors, dim={matrix.shape[1]}")
    return matrix


def embed_one(text: str) -> np.ndarray:
    """Embed a single string. Returns a (1536,) numpy array. Used for queries."""
    return embed_texts([text])[0]


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Divide each row by its L2 norm so dot product == cosine similarity."""
    if matrix.ndim == 1:
        norm = np.linalg.norm(matrix)
        return matrix if norm == 0 else matrix / norm
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


if __name__ == "__main__":
    sample = [
        "What is RAG?",
        "How do I install the WidgetMax 3000?",
    ]
    vecs = embed_texts(sample)
    print("[embeddings] shape:", vecs.shape)
