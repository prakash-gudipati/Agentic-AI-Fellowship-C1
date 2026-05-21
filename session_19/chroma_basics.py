"""
Session 19 — Part A: ChromaDB Basics
─────────────────────────────────────
Goal: build the smallest possible vector database. Persist it.
Add 12 sentences. Run 3 queries.

This is the same engine you wrote in S18 from scratch — except:
  - the loop is a fast index built by Chroma
  - everything saves to disk
  - the API is one line per operation

Run:
    $ pip install chromadb
    $ python chroma_basics.py

PROD PATTERN — PersistentClient (saves to disk).  Same embedding model
end-to-end (Chroma defaults to all-MiniLM-L6-v2 — keep that, don't mix).
"""

import os
import chromadb

# ─────────────────────────────────────────────────────────────
# CONSTANTS — at the top, easy to change
# ─────────────────────────────────────────────────────────────
DB_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")
COLLECTION     = "session19_animals"
TOP_K          = 3

# 12 short sentences — same shape as S18, four loose topics
CORPUS = [
    ("animal_1",  "A loyal dog wags its tail at the door."),
    ("animal_2",  "Puppies grow up to be the most loving pets."),
    ("animal_3",  "Wolves hunt together in coordinated packs."),
    ("vehicle_1", "Modern cars run on electricity instead of petrol."),
    ("vehicle_2", "Aeroplanes cross continents in just a few hours."),
    ("vehicle_3", "Trucks transport goods across long highways."),
    ("food_1",    "Fresh pizza tastes best straight from a wood-fired oven."),
    ("food_2",    "A homemade burger beats any fast-food chain."),
    ("food_3",    "Spaghetti is the world's most comforting pasta dish."),
    ("emotion_1", "Joy spreads quickly when shared with friends."),
    ("emotion_2", "Fear grips the chest before a difficult conversation."),
    ("emotion_3", "Quiet contentment is the underrated form of happiness."),
]


# ─────────────────────────────────────────────────────────────
def build_collection() -> chromadb.api.models.Collection.Collection:
    """Create / load a persistent collection on disk."""
    print(f"  [Step 1/3] Connecting to persistent Chroma at {DB_PATH} ...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLLECTION)
    print(f"             Collection '{COLLECTION}' ready. "
          f"Existing items: {collection.count()}")
    return collection


def populate(collection) -> None:
    """Insert the corpus. Idempotent — uses upsert so re-runs are safe."""
    print(f"\n  [Step 2/3] Upserting {len(CORPUS)} documents...")
    ids       = [item[0] for item in CORPUS]
    documents = [item[1] for item in CORPUS]
    collection.upsert(ids=ids, documents=documents)
    print(f"             Done. Total items in collection: {collection.count()}")


def query(collection, text: str, k: int = TOP_K) -> None:
    """Run one nearest-neighbour query. Print top-K results."""
    print(f"\n  QUERY: {text!r}")
    print("  " + "─" * 60)
    results = collection.query(query_texts=[text], n_results=k)
    # Chroma returns results as lists-of-lists (one per query string).
    # We sent one query, so we read the first slot.
    for rank, (doc_id, doc, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0]), start=1):
        print(f"    [{rank}]  dist = {dist:+.4f}   {doc_id:10s}  {doc}")


def main() -> None:
    print("=" * 64)
    print("  SESSION 19 · PART A — CHROMA BASICS")
    print("=" * 64)

    collection = build_collection()
    populate(collection)

    print("\n  [Step 3/3] Running 3 example queries...")
    queries = [
        "loyal pet that barks",
        "fast travel between continents",
        "comforting home-cooked meal",
    ]
    for q in queries:
        query(collection, q)

    print("\n  ──")
    print(f"  Vectors saved to: {DB_PATH}")
    print("  Run this file again — corpus loads from disk, no re-embedding.")
    print("  This is what a vector database gives you that a Python list cannot.")


if __name__ == "__main__":
    main()
