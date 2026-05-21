# Session 22 — Retrieval Techniques I — Similarity, Filtering, MMR

Phase 4, Session 3 of 8. We tune **STAGE 4** of the RAG pipeline — retrieval —
by introducing three retrieval strategies and a comparison harness that shows
when each one wins.

## What's new vs S20 / S21

- **S20** had ONE retriever — top-K cosine — and one chunker (fixed-char).
- **S21** introduced FIVE chunkers behind a common `Chunker` interface (the
  strategy pattern).
- **S22** introduces THREE retrievers behind a common `Retriever` interface —
  the *retriever* strategy pattern. Plus chunk-level **metadata** that the
  filter retriever uses to narrow the candidate pool before similarity runs.

| Strategy | Picks chunks by | Wins when |
|---|---|---|
| `similarity` | top-K cosine over all chunks (the S20 baseline) | small clean corpus |
| `filtered`   | metadata pre-filter, then top-K cosine | multi-source corpus where you can name what to keep |
| `mmr`        | Maximal Marginal Relevance — relevance balanced against diversity | corpus has near-duplicate chunks |

## Files

| File | What it is |
|------|------------|
| `loaders.py` | Loads PDF / HTML / .txt and attaches per-doc metadata (`source_type`, `section`). Extends the S20 loader. |
| `chunker.py` | Slim wrapper around the S21 recursive chunker. Each `Chunk` now carries `metadata` copied from its parent doc. |
| `embeddings.py` | OpenAI embeddings — unchanged from S20. |
| `retrievers/base.py` | Abstract `Retriever` + `Retrieved` dataclass. The retriever strategy interface. |
| `retrievers/similarity.py` | Pure top-K cosine. The S20 baseline ported. |
| `retrievers/filtered.py` | Metadata pre-filter, then top-K cosine. |
| `retrievers/mmr.py` | MMR — greedy diversity selection with λ knob. |
| `retrievers/__init__.py` | `STRATEGIES = {"similarity": ..., "filtered": ..., "mmr": ...}` |
| `test_queries.py` | 6 queries — each tagged with expected source / metadata filter / "redundancy expected?" flag. |
| `compare.py` | Runs all three retrievers on every test query and prints a side-by-side table. |
| `data/` | The Phase 4 corpus from S20 — copy or symlink it in. |

## Quickstart

```
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Bring the S20 corpus in:
cp -r ../../Session_20/Code/data .

# Smoke-test one retriever:
python -m retrievers.similarity "What is RAG?"

# Compare all three on the full query set:
python compare.py
```

## Expected output (abridged)

```
QUERY  "What is RAG and why does it matter?"
─────────────────────────────────────────────────────────────────────────────
similarity   ✓  intro_to_rag.pdf  intro_to_rag.pdf  intro_to_rag.pdf
filtered     ✓  intro_to_rag.pdf  intro_to_rag.pdf
mmr          ✓  intro_to_rag.pdf  sample_article.html  product_manual.pdf

QUERY  "How do I install the WidgetMax 3000?"   filter={'source_type': 'manual'}
─────────────────────────────────────────────────────────────────────────────
similarity   ✓  product_manual.pdf  product_manual.pdf  intro_to_rag.pdf
filtered     ✓  product_manual.pdf  product_manual.pdf
mmr          ✓  product_manual.pdf  product_manual.pdf  sample_article.html
...
SUMMARY                  hits   diversity
  similarity              5/6        low
  filtered                6/6        med
  mmr                     5/6       high
```

The **shape of the result** is the lesson — pure similarity often returns
near-duplicates from the same source; filtering gives a precision boost when
you know what kind of doc you want; MMR diversifies the top-K when you don't
know which sub-topic the user actually asked about.

## What's NEXT (S23)

Hybrid search (BM25 + dense) and a cross-encoder reranker. After S23 you'll
have five swappable retrievers and the 5×5 chunker × retriever matrix
exercise — the signature Phase 4 benchmark.
