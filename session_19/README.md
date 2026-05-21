# Session 19 — Vector Databases (ChromaDB + FAISS)

Four demo scripts. Each one is independent. Run in order.

## Setup (5 minutes)

```bash
cd Session_19/Code
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run order (matches the live demo)

| # | File | What it shows |
|---|------|---------------|
| 1 | `chroma_basics.py` | Persistent client. Add 12 docs. Query top-3. |
| 2 | `chroma_metadata.py` | Same docs + topic / source tags. `where={...}` filter before similarity. |
| 3 | `faiss_basics.py` | Same docs in FAISS. IndexFlatL2 (exact) + IndexHNSWFlat (ANN). |
| 4 | `compare_chroma_vs_faiss.py` | Time 100 queries on each. |

```bash
python chroma_basics.py
python chroma_metadata.py
python faiss_basics.py
python compare_chroma_vs_faiss.py
```

## What you should see

- A `db/` folder appears next to your scripts — that's the persistent ChromaDB.
- A `faiss_index/flat.faiss` file appears — that's a FAISS index saved to disk (you saved it manually; Chroma did it automatically).
- All four scripts return the same dog-related top match for the query "loyal pet that barks".
- Re-running any script is fast because Chroma loads from disk; the corpus does not get re-embedded.

## Production patterns demonstrated

- **PersistentClient** — vectors survive restart. The default for any production system.
- **Same embedding model end-to-end** — Chroma's `SentenceTransformerEmbeddingFunction` is configured with the same `all-MiniLM-L6-v2` that FAISS uses, so the comparison is fair.
- **Metadata filtering** — narrow the search space BEFORE similarity. Cheap. Always on.
- **Upsert (not insert)** — re-runs are idempotent.
- **Pipeline-style logging** — `[Step N/M]` prefixes throughout.

## Cost note

Everything in Session 19 runs LOCAL — `sentence-transformers` is free, FAISS is free, ChromaDB is free.
No API key required for any of these scripts.

## Reading list

- ChromaDB docs · https://docs.trychroma.com
- FAISS docs · https://faiss.ai
- HNSW paper (the algorithm behind `IndexHNSWFlat`) · Malkov & Yashunin, 2016
