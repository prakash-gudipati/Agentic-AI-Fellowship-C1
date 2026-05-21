# Session 21 — Chunking Strategies Deep Dive

Phase 4, Session 2 of 8. We measure five chunking strategies on the same Phase 4
corpus students built in S20 and learn to pick the right one for the domain.

## What's new vs S20

S20 had ONE chunker — fixed-character. S21 adds four more, all behind a common
`Chunker` interface (the **strategy pattern**), plus a benchmark that scores
each strategy against a fixed test-query set.

## Files

| File | What it is |
|------|------------|
| `chunkers/base.py` | `Chunk` dataclass + abstract `Chunker` base. |
| `chunkers/fixed_char.py` | S20 baseline ported to the new interface. |
| `chunkers/sentence.py` | Sentence-aware chunker. NLTK fallback regex. |
| `chunkers/recursive.py` | Paragraph → sentence → word. The LangChain default. |
| `chunkers/semantic.py` | Embedding-similarity boundary detection. |
| `chunkers/structure.py` | Markdown / HTML heading-aware. |
| `chunkers/__init__.py` | Registry: `STRATEGIES = {"fixed_char": ..., ...}` |
| `benchmark.py` | Runs all 5 over the corpus, scores by Hit Rate. |
| `test_queries.py` | 6 queries with ground-truth source filenames. |
| `requirements.txt` | Pinned deps including `nltk`. |
| `.env.example` | API key placeholders. |

## Reuse from S20

This session **deliberately reuses** Session 20's `loaders.py`, `embeddings.py`,
`retriever.py`, and `generator.py`. Only the chunker is swapped. That's the
strategy pattern in action.

When you run `benchmark.py`, it expects to find Session 20's `loaders.py` and
`embeddings.py` on the import path. Run it from `Session_21/Code/` after
copying or symlinking those files in (or simply running from a combined repo).

## Quickstart

```
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (You should already have your S20 corpus in data/ — copy from Session_20/Code/data/)
cp -r ../../Session_20/Code/data .
cp ../../Session_20/Code/loaders.py .
cp ../../Session_20/Code/embeddings.py .
cp ../../Session_20/Code/retriever.py .

python -c "import nltk; nltk.download('punkt_tab')"

python benchmark.py
```

## Expected output (abridged)

```
Strategy           chunks  hit@3   per-query
fixed_char            18    33%   [✗ ✓ ✗ ✗ ✓ ✗]
sentence              22    50%   [✓ ✓ ✗ ✗ ✓ ✗]
recursive             20    83%   [✓ ✓ ✓ ✓ ✓ ✗]
semantic              16    83%   [✓ ✓ ✓ ✓ ✓ ✗]
structure             14   100%   [✓ ✓ ✓ ✓ ✓ ✓]
```

Numbers will differ on your machine — the **shape of the result** (recursive
and structure beat fixed-char by a wide margin) is the lesson.
