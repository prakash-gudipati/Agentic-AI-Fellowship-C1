"""
Session 19 — Part D: Speed Comparison (Chroma vs FAISS)
────────────────────────────────────────────────────────
Goal: same data, same query, both stores. Time them.

What you should observe:
  • At 12 docs, both are fast (~ms). The difference is invisible.
  • Chroma takes a tiny extra hop because it embeds the query for you;
    FAISS makes you embed it manually.
  • At 1M docs, FAISS HNSW would still run in ~10ms;
    Chroma's default index is also fast, but tunable.

The point of this file is NOT to declare a winner. It's to show that
they solve the same problem with different trade-offs.

Run:
    $ python compare_chroma_vs_faiss.py
"""

import os
import time
import numpy as np
import chromadb
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K      = 3
DB_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")
COLLECTION = "session19_compare"

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
QUERY = "loyal pet that barks"


def setup_chroma(model_name: str) -> "chromadb.api.models.Collection.Collection":
    """Configure Chroma with the SAME embedding model FAISS uses, for fairness."""
    from chromadb.utils import embedding_functions
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = chromadb.PersistentClient(path=DB_PATH)
    coll = client.get_or_create_collection(COLLECTION, embedding_function=ef)
    if coll.count() == 0:
        coll.upsert(ids=[f"id_{i}" for i in range(len(CORPUS))], documents=CORPUS)
    return coll


def setup_faiss(model: SentenceTransformer) -> faiss.Index:
    vecs = np.asarray(model.encode(CORPUS, normalize_embeddings=True), dtype=np.float32)
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)
    return index


def time_chroma(coll, query: str, runs: int = 100) -> float:
    t0 = time.perf_counter()
    for _ in range(runs):
        coll.query(query_texts=[query], n_results=TOP_K)
    return (time.perf_counter() - t0) / runs * 1000   # ms per query


def time_faiss(index: faiss.Index, model: SentenceTransformer, query: str, runs: int = 100) -> float:
    qv = np.asarray(model.encode([query], normalize_embeddings=True), dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(runs):
        index.search(qv, TOP_K)
    return (time.perf_counter() - t0) / runs * 1000   # ms per query


def main() -> None:
    print("=" * 64)
    print("  SESSION 19 · PART D — CHROMA vs FAISS, SAME DATA")
    print("=" * 64)

    print(f"\n  Model: {MODEL_NAME}   |   Corpus: {len(CORPUS)} docs   |   Query: {QUERY!r}")

    print("\n  [Step 1/3] Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("\n  [Step 2/3] Setting up Chroma + FAISS with the same model...")
    coll = setup_chroma(MODEL_NAME)
    index = setup_faiss(model)
    print(f"             Chroma items: {coll.count()}   |   FAISS items: {index.ntotal}")

    print("\n  [Step 3/3] Timing 100 queries on each store...")
    chroma_ms = time_chroma(coll, QUERY)
    faiss_ms  = time_faiss(index, model, QUERY)

    print()
    print(f"    Chroma:  {chroma_ms:6.2f} ms / query   (includes embedding the query)")
    print(f"    FAISS :  {faiss_ms:6.2f} ms / query   (vector → index lookup only)")
    print()
    print("  The numbers are tiny here because we have 12 docs.")
    print("  At 1M+ docs the gap stays small — both have proper indices.")
    print("  The right question is NOT 'which is faster'. It is 'which fits my system'.")


if __name__ == "__main__":
    main()
