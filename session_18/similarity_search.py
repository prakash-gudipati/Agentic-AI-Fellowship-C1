"""
Session 18 — Part B + Part C: Cosine Similarity From Scratch + Top-K Search
──────────────────────────────────────────────────────────────────────────
Goal: Build a tiny semantic-search engine in pure NumPy.
  1. Embed every sentence in our corpus.
  2. Compute cosine similarity between a query and every sentence.
  3. Sort. Return the top-K closest matches.

This is exactly what ChromaDB does under the hood — only this version
fits on one screen and you can read every line.

Run this file:
    $ python similarity_search.py

PROD PATTERN — API key via env var.
PROD PATTERN — Same embedding model used for the corpus AND the query.
PROD PATTERN — Pipeline-style logging with [Step N] prefixes.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 3   # how many "best matches" to return for each query

# Our 12-sentence corpus. Four loose topics, three sentences each:
# ANIMALS · VEHICLES · FOODS · EMOTIONS.
CORPUS: list[str] = [
    # ANIMALS
    "A loyal dog wags its tail at the door.",
    "Puppies grow up to be the most loving pets.",
    "Wolves hunt together in coordinated packs.",
    # VEHICLES
    "Modern cars run on electricity instead of petrol.",
    "Aeroplanes cross continents in just a few hours.",
    "Trucks transport goods across long highways.",
    # FOODS
    "Fresh pizza tastes best straight from a wood-fired oven.",
    "A homemade burger beats any fast-food chain.",
    "Spaghetti is the world's most comforting pasta dish.",
    # EMOTIONS
    "Joy spreads quickly when shared with friends.",
    "Fear grips the chest before a difficult conversation.",
    "Quiet contentment is the underrated form of happiness.",
]


# ─────────────────────────────────────────────────────────────
# Step 1: Embedding helpers
# ─────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    """Load .env and return an OpenAI client."""
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Add it to .env.")
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed a list of strings in ONE API call. Return shape (n, 1536) numpy array."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Step 2: COSINE SIMILARITY — from scratch, six lines
# ─────────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.

        cos(A, B) = (A · B) / (|A| × |B|)

    Returns a number between -1.0 and 1.0:
        +1.0  → vectors point the same way (identical meaning)
         0.0  → vectors are perpendicular (unrelated meaning)
        -1.0  → vectors point opposite ways (opposite meaning)
    """
    dot_product = float(np.dot(a, b))                        # A · B
    length_a = float(np.linalg.norm(a))                      # |A|
    length_b = float(np.linalg.norm(b))                      # |B|
    if length_a == 0.0 or length_b == 0.0:
        return 0.0                                           # avoid divide-by-zero
    return dot_product / (length_a * length_b)


# ─────────────────────────────────────────────────────────────
# Step 3: Top-K search — sort, then slice
# ─────────────────────────────────────────────────────────────
@dataclass
class SearchResult:
    rank: int
    score: float
    text: str


def search(query: str, corpus_texts: list[str], corpus_vectors: np.ndarray,
           query_vector: np.ndarray, k: int = TOP_K) -> list[SearchResult]:
    """Return the top-K most similar sentences to the query."""
    scores = [cosine_similarity(query_vector, vec) for vec in corpus_vectors]
    # Pair each score with its sentence, sort descending, take the top k.
    ranked = sorted(zip(scores, corpus_texts), reverse=True)
    return [
        SearchResult(rank=i + 1, score=score, text=text)
        for i, (score, text) in enumerate(ranked[:k])
    ]


# ─────────────────────────────────────────────────────────────
# Step 4: Compare three distance metrics (bonus — Part E in the demo)
# ─────────────────────────────────────────────────────────────
def dot_product_score(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compare_metrics(query_vector: np.ndarray, corpus_texts: list[str],
                    corpus_vectors: np.ndarray) -> None:
    """Print top-3 results under each of the three distance metrics."""
    print("\n  COMPARING THREE METRICS  (same query, same data, different rankings)")
    for label, fn, reverse in [
        ("COSINE",     cosine_similarity, True),
        ("DOT",        dot_product_score, True),
        ("EUCLIDEAN",  euclidean_distance, False),  # Euclidean: smaller is better
    ]:
        scored = [(fn(query_vector, vec), text) for vec, text in zip(corpus_vectors, corpus_texts)]
        scored.sort(reverse=reverse)
        print(f"\n  --- {label} ---")
        for i, (score, text) in enumerate(scored[:TOP_K], start=1):
            print(f"    {i}. {score:+.4f}  |  {text}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run_query(client: OpenAI, query: str, corpus_vectors: np.ndarray) -> None:
    """Embed the query, run top-K search, print results."""
    print(f"\n  QUERY: {query!r}")
    print("  " + "─" * 60)
    query_vector = embed_texts(client, [query])[0]
    results = search(query, CORPUS, corpus_vectors, query_vector)
    for r in results:
        print(f"    [{r.rank}]  cos = {r.score:+.4f}   {r.text}")


def main() -> None:
    print("=" * 64)
    print("  SESSION 18 · PART B+C — COSINE FROM SCRATCH + TOP-K SEARCH")
    print("=" * 64)

    client = get_client()

    print(f"\n  [Step 1/3] Embedding {len(CORPUS)} corpus sentences...")
    corpus_vectors = embed_texts(client, CORPUS)
    print(f"             Done. Shape = {corpus_vectors.shape}")

    # Quick sanity check — show that two sentences from the same topic
    # have HIGH cosine similarity, and two from different topics have LOW.
    print("\n  [Step 2/3] Sanity checking cosine similarity...")
    pair_close   = cosine_similarity(corpus_vectors[0], corpus_vectors[1])   # both ANIMALS
    pair_far     = cosine_similarity(corpus_vectors[0], corpus_vectors[3])   # ANIMAL vs VEHICLE
    pair_emotion = cosine_similarity(corpus_vectors[0], corpus_vectors[9])   # ANIMAL vs EMOTION
    print(f"             dog-vs-puppy        cos = {pair_close:+.4f}   (expect HIGH)")
    print(f"             dog-vs-electric-car cos = {pair_far:+.4f}   (expect LOW)")
    print(f"             dog-vs-joy          cos = {pair_emotion:+.4f}   (expect LOW)")

    # Real queries — what students will actually run.
    print("\n  [Step 3/3] Running semantic search on 3 example queries...")
    queries = [
        "loyal pet that barks",
        "fast travel between countries",
        "comforting home-cooked meal",
    ]
    for q in queries:
        run_query(client, q, corpus_vectors)

    # Bonus: show how rankings change when you swap the metric.
    sample_query_vec = embed_texts(client, [queries[0]])[0]
    compare_metrics(sample_query_vec, CORPUS, corpus_vectors)

    print()
    print("  Done. You just built the engine inside ChromaDB.")
    print("  Next file: visualize_embeddings.py — see meaning-space in 2D.")


if __name__ == "__main__":
    main()
