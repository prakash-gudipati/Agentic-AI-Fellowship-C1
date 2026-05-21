# Session 20 — RAG Architecture End-to-End

This is the reference build for **Phase 4, Session 20** of the Agentic AI Builders Fellowship.
We build the **simplest possible end-to-end RAG pipeline in pure Python** over a real, messy corpus
(2 PDFs + 1 webpage HTML). No vector database. No framework. Just the 5 stages, naked.

> Mental model before optimisation. Once you feel each stage doing its job,
> Sessions 21–28 will tune every one of them.

---

## The 5 stages we build

```
[1] INGEST     →  Load PDF / HTML / .txt files into clean text
[2] CHUNK      →  Slice text into ~500-char Post-it notes (with overlap)
[3] EMBED      →  Turn each chunk into a dense vector (OpenAI embeddings)
[4] RETRIEVE   →  Cosine top-K against the user's question
[5] GENERATE   →  Stuff retrieved chunks into a grounded prompt → Claude
```

---

## Files

| File | What it is |
|------|------------|
| `data/intro_to_rag.pdf` | Sample 2-page research-style PDF. |
| `data/product_manual.pdf` | Sample 2-page product manual PDF. |
| `data/sample_article.html` | Sample webpage with messy nav/footer markup. |
| `generate_corpus.py` | Builds the 2 PDFs + HTML from text. Run once. |
| `loaders.py` | PDF / HTML / .txt → clean text. |
| `chunker.py` | Character-based chunker with overlap. |
| `embeddings.py` | OpenAI embeddings wrapper. |
| `retriever.py` | In-memory store + cosine top-K. |
| `generator.py` | Grounded prompt template + Claude call. |
| `rag_pipeline.py` | The 5 stages wired end-to-end. CLI entry-point. |
| `requirements.txt` | Pinned dependencies. |
| `.env.example` | API key placeholders. |

---

## Quickstart

```
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your real keys

python generate_corpus.py          # build the sample data once
python rag_pipeline.py "What is RAG and why does it matter?"
python rag_pipeline.py "How do I install the WidgetMax 3000?"
python rag_pipeline.py "What is the capital of France?"   # out-of-corpus → "I don't know"
```

---

## What is intentionally NOT here

- **No vector database.** We use a Python list. Phase 4's later sessions add ChromaDB.
- **No reranking.** Top-K cosine only. Reranking comes in S23.
- **No hybrid search.** Pure dense retrieval. Hybrid comes in S23.
- **No evaluation framework.** Eyeball the answers today. Ragas comes in S25.
- **No observability.** Print logs only. LangSmith comes in S27.

This is the *mental model*. Optimisation is what the rest of Phase 4 is for.
