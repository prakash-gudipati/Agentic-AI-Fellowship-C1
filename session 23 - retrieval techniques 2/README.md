# Session 23 — Retrieval Techniques II — Hybrid Search + Reranking

Phase 4 · Session 4 of 8 · 80 min · Standard Technical

This session adds the last two retrieval strategies to the toolkit from S22
and introduces the 25-combination matrix workflow.

## What's new vs. Session 22

| File | Purpose |
|---|---|
| `retrievers/bm25.py` | Pure-Python BM25 — sparse keyword scoring |
| `retrievers/hybrid.py` | Dense + sparse, two fusion modes (weighted, RRF) |
| `retrievers/rerank.py` | LLM-as-judge reranker (Claude Haiku 4.5) — wraps any base retriever |
| `chunkers.py` | 5 chunkers behind one interface, for the matrix |
| `matrix.py` | The signature demo — 5 chunkers × 5 retrievers, hit-rate per cell |
| `compare.py` | Extended S22 compare — now runs all 5 retrievers side-by-side |

Everything else (loaders, embeddings, base retriever, ChunkIndex, S22's
three retrievers) is unchanged.

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # then add your keys
```

You need an OPENAI_API_KEY for dense retrieval (embeddings) and an
ANTHROPIC_API_KEY for the rerank column.

## Run the comparisons

```bash
# All five retrievers, fixed chunker, 8 test queries:
python compare.py

# The signature 5×5 matrix:
python matrix.py

# Skip the LLM column if no Anthropic key:
python matrix.py --skip-rerank

# Skip semantic chunker if you don't want to pay for sentence embeddings:
python matrix.py --skip-rerank --skip-semantic
```

## The five retrievers

| Strategy | Signal | When it wins | When it loses |
|---|---|---|---|
| `similarity` | dense cosine | small clean corpus, conceptual queries | rare exact terms, near-duplicates |
| `bm25` | sparse keyword (TF·IDF + length norm) | product codes, error numbers, names | paraphrased queries, synonyms |
| `hybrid` | weighted dense + sparse, or RRF | most general-purpose RAG | when one signal is much stronger than the other |
| `mmr` | similarity with diversity penalty | redundant corpora | when diversity is irrelevant |
| `rerank` | first-pass + LLM-as-judge | nuanced, paraphrased queries | latency-sensitive, cost-sensitive |

## The 25-combination matrix — what it is for

`matrix.py` is not a one-time test. It is the WORKFLOW you run every time
you change your corpus, your queries, or your chunking strategy. The same
chunker that wins for FAQ docs loses for long manuals. Measure, then pick.

## Production patterns named in this session

- **Score normalisation before fusion** — min-max per query, then α-blend.
- **Two-stage retrieval** — cheap recall first, expensive precision second.
- **Measure before you guess** — the matrix workflow itself.
- Retriever strategy pattern (reinforced from S22).
- LLM-as-judge (callback to S14, applied inside retrieval).
