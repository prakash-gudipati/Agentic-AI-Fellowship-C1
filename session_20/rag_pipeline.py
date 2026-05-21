"""
Session 20 — rag_pipeline.py

The 5 stages wired end-to-end + a CLI entry-point.

Run:
    python rag_pipeline.py "What is RAG and why does it matter?"
    python rag_pipeline.py "How do I install the WidgetMax 3000?"
    python rag_pipeline.py "What is the capital of France?"   # out-of-corpus

The pipeline runs INDEX-TIME work (load → chunk → embed → store) on first call,
then QUERY-TIME work (retrieve → generate) for every subsequent call.

Production patterns reinforced:
  - pipeline logging with stage numbers — [1/5], [2/5], etc.
  - try/except on every external call inside loaders/embeddings/generator
  - env vars for secrets
  - clear separation between index-time and query-time code paths
"""

import os
import sys
from typing import List

from loaders import load_corpus
from chunker import chunk_corpus
from retriever import InMemoryStore
from generator import generate

# ── Configuration ───────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = [
    os.path.join(HERE, "data", "intro_to_rag.pdf"),
    os.path.join(HERE, "data", "product_manual.pdf"),
    os.path.join(HERE, "data", "sample_article.html"),
]
TOP_K = 3


def build_index(paths: List[str]) -> InMemoryStore:
    """INDEX-TIME pipeline: stages 1-3.

    Slow. Runs once per corpus update.
    """
    print("\n" + "═" * 70)
    print("INDEX-TIME — runs once when the corpus changes")
    print("═" * 70)

    print("\n[1/5] INGEST  — loading documents from disk")
    docs = load_corpus(paths)

    print("\n[2/5] CHUNK   — slicing each document into ~500-char pieces")
    chunks = chunk_corpus(docs)

    print("\n[3/5] EMBED   — turning each chunk into a dense vector")
    store = InMemoryStore()
    store.add(chunks)

    print(f"\nINDEX-TIME complete — {len(store.chunks)} chunks ready for search\n")
    return store


def answer_question(store: InMemoryStore, question: str, k: int = TOP_K) -> str:
    """QUERY-TIME pipeline: stages 4-5.

    Fast. Runs once per user question.
    """
    print("═" * 70)
    print(f"QUERY-TIME — answering: {question!r}")
    print("═" * 70)

    print("\n[4/5] RETRIEVE — top-K most-similar chunks")
    retrieved = store.search(question, k=k)
    for r in retrieved:
        preview = r.chunk.text[:80].replace("\n", " ")
        print(f"          score={r.score:.3f}  [{r.chunk.source}]  {preview}...")

    print("\n[5/5] GENERATE — grounded prompt → Claude")
    answer = generate(question, retrieved)
    return answer


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python rag_pipeline.py \"your question\"")
        return 1
    question = " ".join(sys.argv[1:])

    # Verify the sample corpus exists. If not, prompt the student to build it.
    missing = [p for p in DEFAULT_CORPUS if not os.path.exists(p)]
    if missing:
        print("ERROR: sample corpus is missing the following files:")
        for p in missing:
            print("  -", p)
        print("\nRun this first:    python generate_corpus.py")
        return 1

    store = build_index(DEFAULT_CORPUS)
    answer = answer_question(store, question)

    print("\n" + "─" * 70)
    print("ANSWER")
    print("─" * 70)
    print(answer)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
