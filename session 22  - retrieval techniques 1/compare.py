"""
Session 22 — compare.py

The session's signature demo. Builds ONE chunk index, then runs all THREE
retrievers — similarity, filtered, mmr — over the same fixed test query set.
Prints a side-by-side table so the differences jump out.

Usage:
    python compare.py                 # default K=3
    python compare.py --k 5
    python compare.py --lambda 0.3    # bias MMR toward diversity

Reads from data/. Bring the S20 corpus in first:
    cp -r ../../Session_20/Code/data .

Production patterns shown:
  - retriever strategy pattern — ONE index shared by THREE retrievers
  - pre-filter before scoring — filter retriever
  - balance relevance with diversity — mmr retriever
  - pipeline logging with stage numbers — INDEX-TIME vs QUERY-TIME phases
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

# Make sibling modules importable when running as a script.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from loaders import load_corpus
from chunker import chunk_corpus
from retrievers import STRATEGIES
from retrievers.base import ChunkIndex, Retrieved
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


def run_one(retriever, q: TestQuery, k: int, lambda_: float) -> RowResult:
    """Run ONE retriever on ONE query. Score by hit + distinct-source count."""
    if retriever.name == "filtered":
        results = retriever.pick(q.question, k=k, filter=q.filter)
    elif retriever.name == "mmr":
        results = retriever.pick(q.question, k=k, lambda_=lambda_)
    else:
        results = retriever.pick(q.question, k=k)

    sources = [r.chunk.source for r in results]
    distinct = len(set(sources))
    hit = q.expected_source in sources
    return RowResult(strategy=retriever.name, hit=hit,
                     sources=sources, distinct_sources=distinct)


def print_query_block(q: TestQuery, rows: List[RowResult]) -> None:
    print()
    print("─" * 78)
    flt = f"  filter={q.filter}" if q.filter else ""
    print(f"QUERY  {q.question!r}{flt}")
    print(f"        expect={q.expected_source}  "
          f"redundancy_expected={q.redundancy_expected}")
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
    print(f"  {'Strategy':<12}{'hits':>8}{'rate':>8}{'avg distinct':>16}")
    print("  " + "-" * 42)
    for name, rows in per_strategy.items():
        hits = sum(1 for r in rows if r.hit)
        avg_distinct = sum(r.distinct_sources for r in rows) / max(len(rows), 1)
        print(f"  {name:<12}{hits:>4}/{total:<3}{hits / total * 100:>7.0f}%"
              f"{avg_distinct:>16.2f}")
    print()
    print("  Reading the table:")
    print("    hits         — did the expected source appear in top-K?")
    print("    avg distinct — average number of *different* sources in top-K")
    print()
    print("  No 'best retriever' exists in general — match the strategy to the")
    print("  shape of your corpus and your queries.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="S22 retriever comparison")
    parser.add_argument("--k", type=int, default=3,
                        help="top-K for retrieval (default 3)")
    parser.add_argument("--lambda_", "--lambda", dest="lambda_", type=float,
                        default=0.5, help="MMR λ (default 0.5)")
    args = parser.parse_args()

    missing = [p for p in DEFAULT_CORPUS if not os.path.exists(p)]
    if missing:
        print("ERROR: corpus is missing:")
        for p in missing:
            print("  -", p)
        print("\nCopy the S20 corpus first, e.g.:")
        print("  cp -r ../../Session_20/Code/data .")
        return 1

    # ── INDEX-TIME ──────────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("INDEX-TIME — runs once for all three retrievers")
    print("═" * 78)
    print("\n[1/3] INGEST  — loading corpus")
    docs = load_corpus(DEFAULT_CORPUS)

    print("\n[2/3] CHUNK   — recursive chunker (S21 default)")
    chunks = chunk_corpus(docs)

    print("\n[3/3] EMBED   — building shared ChunkIndex")
    index = ChunkIndex()
    index.add(chunks)

    # ── QUERY-TIME ──────────────────────────────────────────────────────────
    retrievers = {name: cls(index) for name, cls in STRATEGIES.items()}

    print("\n" + "═" * 78)
    print(f"QUERY-TIME — comparing {len(retrievers)} retrievers over "
          f"{len(TEST_QUERIES)} queries  (K={args.k}, MMR λ={args.lambda_})")
    print("═" * 78)

    per_strategy: dict = {name: [] for name in retrievers}
    for q in TEST_QUERIES:
        rows = []
        for name, r in retrievers.items():
            row = run_one(r, q, k=args.k, lambda_=args.lambda_)
            per_strategy[name].append(row)
            rows.append(row)
        print_query_block(q, rows)

    print_summary(per_strategy, total=len(TEST_QUERIES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
