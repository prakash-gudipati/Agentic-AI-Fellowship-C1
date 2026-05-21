# Session 24 — Advanced RAG Patterns

Phase 4 · Session 5 of 8 · 80 min · Standard Technical

Four decorator patterns that bolt onto the base RAG system from S20–S23 and
consistently lift retrieval quality without rewriting the pipeline:

| Pattern | Type | Cost | Where it shines |
|---|---|---|---|
| Parent-child | Index-time | +50% index size | Long docs where small chunks match but big chunks answer |
| Contextual retrieval | Index-time | 1 LLM call / chunk (cacheable) | Corpora where chunks lose meaning when separated from their doc |
| HyDE | Query-time | 1 LLM call / query | Well-formed questions where query and document language differ |
| Multi-query | Query-time | 3–5 LLM calls + N retrievals / query | Paraphrase-sensitive corpora, ambiguous queries |

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # paste your OPENAI_API_KEY and ANTHROPIC_API_KEY
```

## Run the demo

```bash
# Fast — skips the contextual indexing cost
python demo.py --skip-contextual

# Full — includes contextual retrieval (one LLM call per chunk at ingest)
python demo.py

# Custom query
python demo.py --query "What does the LED on the WidgetMax mean?"
```

## File layout

```
Session_24/Code/
  data/                              ← corpus (carried from S20–S23)
  loaders.py, embeddings.py, chunker.py   ← unchanged from S23
  retrievers/                        ← S23's five retrievers, unchanged
  patterns/                          ← NEW today
    parent_child.py
    contextual.py
    hyde.py
    multi_query.py
  demo.py                            ← runs every pattern on one query
  test_queries.py                    ← S23 queries plus 2 paraphrased ones
  requirements.txt
  .env.example
```

## Production patterns named in this session

- **Compose, don't replace** — every pattern wraps the base RAG.
- **Index-time investment for query-time gains** — pay once at ingest, save forever.
- **Query rewriting before retrieval** — the query you send to the retriever is rarely the one the user typed.
- Retriever strategy pattern (reinforced from S22).
- RRF for fusion (reinforced from S23).
- Prompt caching (reinforced from S16 — makes contextual retrieval practical).
