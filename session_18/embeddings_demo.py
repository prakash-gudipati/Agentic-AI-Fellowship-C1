"""
Session 18 — Part A: Your First Embedding
─────────────────────────────────────────
Goal: Take a single piece of text. Send it to the OpenAI embeddings API.
Print the vector that comes back. Notice three things:
  1. It is just a list of numbers (a "vector").
  2. The list is exactly 1,536 long (for text-embedding-3-small).
  3. Same input → same output, every time. Deterministic.

Run this file:
    $ python embeddings_demo.py

PROD PATTERN — API key via env var (.env). Never hardcoded.
PROD PATTERN — Same embedding model for ingestion AND queries.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# CONSTANTS — declared at top of file so they're easy to find/change.
# ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"   # 1,536-dim vectors. Cheap. Good enough for everything Phase 3 needs.
PREVIEW_NUMBERS = 8                           # how many of the 1,536 numbers to print


def get_client() -> OpenAI:
    """Load .env and return an OpenAI client. Crashes loudly if the key is missing."""
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Add it to your .env file:\n"
            "    OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=api_key)


def embed_one(client: OpenAI, text: str) -> list[float]:
    """Send a single piece of text to the embeddings API. Return the vector."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    # The API returns a list of embedding objects (one per input).
    # We sent one input, so we read the first object's `.embedding` field.
    return response.data[0].embedding


def show_vector(text: str, vector: list[float]) -> None:
    """Pretty-print a vector with a labelled header. Doesn't drown the terminal."""
    print(f"\n  TEXT     : {text!r}")
    print(f"  LENGTH   : {len(vector)}  (this is the 'dimension' of the embedding)")
    preview = ", ".join(f"{n:+.4f}" for n in vector[:PREVIEW_NUMBERS])
    print(f"  PREVIEW  : [{preview}, ...]")
    print(f"  TYPE     : {type(vector).__name__} of {type(vector[0]).__name__}")


def main() -> None:
    print("=" * 64)
    print("  SESSION 18 · PART A — YOUR FIRST EMBEDDING")
    print("=" * 64)

    client = get_client()

    # Three short pieces of text. Same model. Same shape of output.
    samples = [
        "A loyal dog barks at strangers.",
        "Puppies bring joy to every household.",
        "An aeroplane flies across continents.",
    ]

    for sample in samples:
        vector = embed_one(client, sample)
        show_vector(sample, vector)

    print()
    print("  Notice: every vector has the SAME length (1,536).")
    print("  Notice: the numbers themselves carry the meaning. We measure")
    print("          'meaning' by measuring how close two vectors are.")
    print("          We will do that in similarity_search.py next.")


if __name__ == "__main__":
    main()
