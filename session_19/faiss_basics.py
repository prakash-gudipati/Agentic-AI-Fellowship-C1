"""
Session 19 — Part C: FAISS Basics
──────────────────────────────────
Goal: same task, different vector store. FAISS is lower-level —
you manage the embeddings yourself, you manage IDs yourself, you
manage save/load yourself. In return: raw speed.

Run:
    $ pip install faiss-cpu sentence-transformers
    $ python faiss_basics.py

Two indices in this file:
  - IndexFlatL2  — exact, brute-force, the simplest possible index
  - IndexHNSWFlat — ANN, much faster on large data, ~99% accuracy
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"        # 384-dim, free, local
TOP_K      = 3
INDEX_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
os.makedirs(INDEX_DIR, exist_ok=True)

CORPUS = [
    "A loyal dog wags its tail at the door.",
    "Puppies grow up to be the most loving pets.",
    "Wolves hunt together in coordinated packs.",
    "Modern cars run on electricity instead of petrol.",
    "Aeroplanes cross continents in just a few hours.",
    "Trucks transport goods across long highways.",
    "Fresh pizza tastes best straight from a wood-fired oven.",
    "A homemade burger beats any fast-food chain.",
    "Spaghetti is the world's most comforting pasta dish.",
    "Joy spreads quickly when shared with friends.",
    "Fear grips the chest before a difficult conversation.",
    "Quiet contentment is the underrated form of happiness.",
]


def embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode strings → matrix of float32 vectors. FAISS needs float32."""
    return np.asarray(model.encode(texts, normalize_embeddings=True),
                      dtype=np.float32)


def build_flat_index(vectors: np.ndarray) -> faiss.Index:
    """IndexFlatL2 — exact, brute-force search. Fine for <100k vectors."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def build_hnsw_index(vectors: np.ndarray) -> faiss.Index:
    """IndexHNSWFlat — ANN graph index. Far faster at scale."""
    dim = vectors.shape[1]
    # M = 32 — neighbours per node in the graph. Higher = more accurate, more RAM.
    index = faiss.IndexHNSWFlat(dim, 32)
    index.add(vectors)
    return index


def search(index: faiss.Index, query_vec: np.ndarray, label: str) -> None:
    distances, ids = index.search(query_vec, TOP_K)
    print(f"\n    {label}")
    for rank, (i, d) in enumerate(zip(ids[0], distances[0]), start=1):
        print(f"        [{rank}]  dist={d:+.4f}   id={i:2d}   {CORPUS[i]}")


def main() -> None:
    print("=" * 64)
    print("  SESSION 19 · PART C — FAISS BASICS")
    print("=" * 64)

    print(f"\n  [Step 1/4] Loading local model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("\n  [Step 2/4] Embedding 12 sentences...")
    vectors = embed(model, CORPUS)
    print(f"             Shape: {vectors.shape}   dtype: {vectors.dtype}")

    print("\n  [Step 3/4] Building two indices...")
    flat = build_flat_index(vectors)
    hnsw = build_hnsw_index(vectors)
    print(f"             IndexFlatL2:    {flat.ntotal} vectors")
    print(f"             IndexHNSWFlat:  {hnsw.ntotal} vectors")

    print("\n  [Step 4/4] Same query on both indices:")
    query_text = "loyal pet that barks"
    print(f"             QUERY: {query_text!r}")
    query_vec = embed(model, [query_text])

    search(flat, query_vec, "FLAT  (exact)")
    search(hnsw, query_vec, "HNSW  (ANN)")

    # Save the FLAT index to disk — FAISS persistence is one function call.
    flat_path = os.path.join(INDEX_DIR, "flat.faiss")
    faiss.write_index(flat, flat_path)
    print(f"\n  Saved FLAT index: {flat_path}")
    print("  Note: persistence is YOUR responsibility in FAISS — Chroma does it for free.")


if __name__ == "__main__":
    main()
