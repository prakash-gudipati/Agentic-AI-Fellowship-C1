"""
Session 23 — compare.py

The "5 retrievers, 1 chunker" comparison — extends S22's compare.py with
the two new retrievers (bm25, hybrid) and the reranker.

Reads the recursive chunker (the session-default) so the chunking variable
is held constant; the only thing varying is the retriever. Run matrix.py
when you want to vary BOTH chunker and retriever.

Usage:
    python compare.py                     # default K=3
    python compare.py --k 5
    python compare.py --skip-rerank       # skip the LLM column (no API key)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from loaders import load_corpus
from chunker import chunk_corpus
from retrievers.base import ChunkIndex
from retrievers.similarity import SimilarityRetriever
from retrievers.bm25 import BM25Retriever
from retrievers.hybrid import HybridRetriever
from retrievers.mmr import MMRRetriever
from retrievers.rerank import LLMRerankRetriever
from test_queries import TEST_QUERIES, TestQuery


DEFAULT_CORPUS = [
    os.path.join(HERE, "data", "intro_to_rag.pdf"),
    os.path.join(HERE, "data", "product_manual.pdf"),
    os.path.join(HERE, "data", "sample_article.html"),
]


@dataclass
class RowResult:
    strategy: str
    hit: bool
    sources: List[str]
    distinct_sources: int


def run_one(retriever, q: TestQuery, k: int) -> RowResult:
    """Run ONE retriever on ONE query. Score by hit + distinct-source count."""
    results = retriever.pick(q.question, k=k)
    sources = [r.chunk.source for r in results]
    distinct = len(set(sources))
    hit = q.expected_source in sources
    return RowResult(strategy=retriever.name, hit=hit,
                     sources=sources, distinct_sources=distinct)


def print_query_block(q: TestQuery, rows: List[RowResult]) -> None:
    print()
    print("─" * 78)
    flags = []
    if q.redundancy_expected:
        flags.append("redundancy")
    if q.keyword_expected:
        flags.append("keyword-heavy")
    flag_str = ("  [" + ", ".join(flags) + "]") if flags else ""
    print(f"QUERY  {q.question!r}{flag_str}")
    print(f"        expect={q.expected_source}")
    print("─" * 78)
    for row in rows:
        marker = "✓" if row.hit else "✗"
        srcs = ",  ".join(row.sources) if row.sources else "(none)"
        print(f"  {row.strategy:<11}  {marker}   "
              f"distinct={row.distinct_sources}   {srcs}")


def print_summary(per_strategy: dict, total: int) -> None:
    print()
    print("═" * 78)
    print("  SUMMARY")
    print("═" * 78)
    print(f"  {'Strategy':<12}{'hits':>10}{'rate':>10}{'avg distinct':>16}")
    print("  " + "-" * 46)
    for name, rows in per_strategy.items():
        hits = sum(1 for r in rows if r.hit)
        avg_distinct = sum(r.distinct_sources for r in rows) / max(len(rows), 1)
        print(f"  {name:<12}{hits:>4}/{total:<5}"
              f"{hits / total * 100:>9.0f}%"
              f"{avg_distinct:>16.2f}")
    print()
    print("  No 'best retriever' exists in general — match the strategy to")
    print("  the shape of your corpus and your queries. Run matrix.py for")
    print("  the full 5x5 grid.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="S23 retriever comparison (5 retrievers)")
    parser.add_argument("--k", type=int, default=3,
                        help="top-K for retrieval (default 3)")
    parser.add_argument("--skip-rerank", action="store_true",
                        help="skip the rerank row (no LLM calls)")
    args = parser.parse_args()

    missing = [p for p in DEFAULT_CORPUS if not os.path.exists(p)]
    if missing:
        print("ERROR: corpus is missing:")
        for p in missing:
            print("  -", p)
        return 1

    print("\n" + "═" * 78)
    print("INDEX-TIME — runs once for all retrievers")
    print("═" * 78)
    print("\n[1/3] INGEST  — loading corpus")
    docs = load_corpus(DEFAULT_CORPUS)
    print("\n[2/3] CHUNK   — recursive chunker (S22 default)")
    chunks = chunk_corpus(docs)
    print("\n[3/3] EMBED   — building shared ChunkIndex")
    index = ChunkIndex()
    index.add(chunks)

    print("\n" + "═" * 78)
    print(f"QUERY-TIME — comparing retrievers over "
          f"{len(TEST_QUERIES)} queries  (K={args.k})")
    print("═" * 78)

    retrievers: dict = {
        "similarity": SimilarityRetriever(index),
        "bm25":       BM25Retriever(index),
        "hybrid":     HybridRetriever(index, mode="weighted", alpha=0.5),
        "mmr":        MMRRetriever(index),
    }
    if not args.skip_rerank:
        retrievers["rerank"] = LLMRerankRetriever(index, fetch_k=10)

    per_strategy: dict = {name: [] for name in retrievers}
    for q in TEST_QUERIES:
        rows = []
        for name, r in retrievers.items():
            row = run_one(r, q, k=args.k)
            per_strategy[name].append(row)
            rows.append(row)
        print_query_block(q, rows)

    print_summary(per_strategy, total=len(TEST_QUERIES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
