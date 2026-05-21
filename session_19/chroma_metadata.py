"""
Session 19 — Part B: Metadata Filtering
────────────────────────────────────────
Goal: tag every document with its topic, then filter BEFORE the
similarity search. Compare results with-filter vs without-filter.

Why this matters: at scale, similarity alone returns false matches.
'happy dog' might find 'happy person' before 'loyal dog'. A metadata
filter narrows the search space first — fewer candidates, more precision.

Run:
    $ python chroma_metadata.py

PROD PATTERN — Pre-filter with metadata. Cheap. Always on.
"""

import os
import chromadb

DB_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")
COLLECTION     = "session19_topics"
TOP_K          = 3

# Same 12 sentences, now with topic + source metadata
CORPUS = [
    ("a1",  "A loyal dog wags its tail at the door.",                "animal",  "blog"),
    ("a2",  "Puppies grow up to be the most loving pets.",            "animal",  "magazine"),
    ("a3",  "Wolves hunt together in coordinated packs.",             "animal",  "documentary"),
    ("v1",  "Modern cars run on electricity instead of petrol.",       "vehicle", "news"),
    ("v2",  "Aeroplanes cross continents in just a few hours.",        "vehicle", "magazine"),
    ("v3",  "Trucks transport goods across long highways.",            "vehicle", "blog"),
    ("f1",  "Fresh pizza tastes best straight from a wood-fired oven.","food",    "blog"),
    ("f2",  "A homemade burger beats any fast-food chain.",            "food",    "magazine"),
    ("f3",  "Spaghetti is the world's most comforting pasta dish.",    "food",    "blog"),
    ("e1",  "Joy spreads quickly when shared with friends.",           "emotion", "essay"),
    ("e2",  "Fear grips the chest before a difficult conversation.",   "emotion", "essay"),
    ("e3",  "Quiet contentment is the underrated form of happiness.",  "emotion", "essay"),
]


def build_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    coll = client.get_or_create_collection(COLLECTION)
    return coll


def populate(coll) -> None:
    print(f"\n  [Step 1/3] Upserting {len(CORPUS)} docs with metadata...")
    ids = [c[0] for c in CORPUS]
    docs = [c[1] for c in CORPUS]
    metas = [{"topic": c[2], "source": c[3]} for c in CORPUS]
    coll.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"             Done. Total: {coll.count()}")


def show(label: str, results: dict) -> None:
    """Pretty-print Chroma results. Skips if empty."""
    print(f"\n    {label}")
    if not results["ids"][0]:
        print("        (no matches under filter)")
        return
    for r, (doc_id, doc, dist, meta) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]), start=1):
        print(f"        [{r}]  dist={dist:+.4f}  topic={meta['topic']:8s}  {doc}")


def main() -> None:
    print("=" * 64)
    print("  SESSION 19 · PART B — METADATA FILTERING")
    print("=" * 64)

    coll = build_collection()
    populate(coll)

    query_text = "happy comforting feeling"

    print(f"\n  [Step 2/3] Same query, no filter:    {query_text!r}")
    no_filter = coll.query(query_texts=[query_text], n_results=TOP_K)
    show("WITHOUT FILTER", no_filter)

    print(f"\n  [Step 3/3] Same query, filter to topic='emotion':")
    with_filter = coll.query(
        query_texts=[query_text],
        n_results=TOP_K,
        where={"topic": "emotion"},
    )
    show("WITH FILTER  (topic='emotion')", with_filter)

    # A second example — filter by source
    print("\n  Bonus: filter to source='blog' for query 'tasty meal':")
    blog_only = coll.query(
        query_texts=["tasty meal"],
        n_results=TOP_K,
        where={"source": "blog"},
    )
    show("WITH FILTER  (source='blog')", blog_only)

    print("\n  ──")
    print("  Notice: with the filter, the search space shrank from 12 to ~3 vectors")
    print("  AND the results stay on-topic. That is metadata filtering.")


if __name__ == "__main__":
    main()
