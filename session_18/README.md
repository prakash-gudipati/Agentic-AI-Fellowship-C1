# Session 18 — Embeddings + Similarity Search from Scratch

Three demo scripts you'll see live in class, plus one backup that works
without an API key.

## Setup (5 minutes)

```bash
cd Session_18/Code
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# then open .env and paste your real OPENAI_API_KEY
```

## Run order (matches the live demo)

| # | File | What it shows |
|---|------|---------------|
| 1 | `embeddings_demo.py` | Your first embedding. Text in → vector out. |
| 2 | `similarity_search.py` | Cosine similarity from scratch · top-K search · cosine vs dot vs Euclidean. |
| 3 | `visualize_embeddings.py` | PCA → 2D scatter plot (`embeddings_plot.png`). |
| 4 | `mock_embeddings_demo.py` | **Backup** — runs with a free local model, no API key. |

```bash
python embeddings_demo.py
python similarity_search.py
python visualize_embeddings.py        # writes embeddings_plot.png
python mock_embeddings_demo.py        # writes embeddings_plot_local.png
```

## What you should see

`visualize_embeddings.py` produces a PNG showing 12 sentences as 12 dots
in 2D space. ANIMAL sentences cluster together. VEHICLE sentences cluster
together. FOOD sentences cluster together. EMOTION sentences cluster
together. That clustering is **meaning-space** — embeddings that mean
similar things live near each other.

## Production patterns demonstrated

- API key loaded from `.env`, never hardcoded.
- The **same** embedding model used for both ingestion and queries.
- Pipeline-style logging with `[Step N]` prefixes.
- Constants at the top of each file.
- Single-responsibility functions with one-line docstrings.

## Cost note

`text-embedding-3-small` costs roughly **$0.00002 per 1K tokens**.
The full demo embeds about 100 tokens total — well under one US cent.
