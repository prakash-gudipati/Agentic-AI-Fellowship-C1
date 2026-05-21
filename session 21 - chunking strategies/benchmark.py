"""
Session 21 — benchmark.py

Run all 5 chunking strategies over the same Phase 4 corpus, score each by
HIT RATE (recall@K) over a fixed test query set, and print a comparison table.

Usage:
    python benchmark.py                    # default K=3
    python benchmark.py --k 5              # bump top-K
    python benchmark.py --strategies recursive structure   # subset

Reuses Session 20's loaders, embeddings, and retriever modules.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

# Reuse S20 modules. Either drop them next to this file, or symlink them in
# (see README.md). We add the local dir to sys.path defensively.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunkers import STRATEGIES
from chunkers.base import Chunk
from loaders import load_corpus           # from S20
from retriever import InMemoryStore       # from S20
from test_queries import TEST_QUERIES, TestQuery


# ── Configuration ───────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = [
    os.path.join(HERE, "data", "intro_to_rag.pdf"),
    os.path.join(HERE, "data", "product_manual.pdf"),
    os.path.join(HERE, "data", "sample_article.html"),
]
DEFAULT_K = 3


@dataclass
class StrategyResult:
    name: str
    chunks: int
    hits: int
    total: int
    per_query: List[bool]  # one bool per TEST_QUERIES item

    @property
    def hit_rate(self) -> float:
        return 0.0 if self.total == 0 else self.hits / self.total


def evaluate_strategy(name: str, chunker_cls,
                      docs: List, queries: List[TestQuery], k: int) -> StrategyResult:
    """Build a fresh index with this chunker, run every test query, score it."""
    print()
    print("─" * 70)
    print(f"  STRATEGY  {name}")
    print("─" * 70)

    chunker = chunker_cls()
    chunks = chunker.chunk_corpus(docs)

    store = InMemoryStore()
    store.add(chunks)

    per_query: List[bool] = []
    for q in queries:
        results = store.search(q.question, k=k)
        sources = {r.chunk.source for r in results}
        hit = q.expected_source in sources
        per_query.append(hit)
        marker = "✓" if hit else "✗"
        print(f"    {marker}  {q.question[:54]:<54}  "
              f"expected={q.expected_source:<22}  "
              f"got={sorted(sources)}")

    hits = sum(per_query)
    return StrategyResult(name=name, chunks=len(chunks),
                          hits=hits, total=len(queries),
                          per_query=per_query)


def print_summary(results: List[StrategyResult]) -> None:
    print()
    print("═" * 70)
    print("  SUMMARY")
    print("═" * 70)
    header = f"  {'Strategy':<14}{'chunks':>8}{'hits':>6}{'rate':>8}   per-query"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        marks = " ".join("✓" if h else "✗" for h in r.per_query)
        print(f"  {r.name:<14}{r.chunks:>8}{r.hits:>6}{r.hit_rate * 100:>7.0f}%   [{marks}]")
    print()
    best = max(results, key=lambda r: r.hit_rate)
    worst = min(results, key=lambda r: r.hit_rate)
    print(f"  Best:  {best.name}  ({best.hit_rate * 100:.0f}%)   ·   "
          f"Worst:  {worst.name}  ({worst.hit_rate * 100:.0f}%)")
    print()
    print("  No 'best chunker' exists in general — the right answer depends on "
          "your domain.")
    print("  Run this benchmark on YOUR corpus before picking a strategy.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="S21 chunker benchmark")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"top-K for retrieval (default {DEFAULT_K})")
    parser.add_argument("--strategies", nargs="*",
                        default=list(STRATEGIES.keys()),
                        help="strategies to run (default: all)")
    args = parser.parse_args()

    missing = [p for p in DEFAULT_CORPUS if not os.path.exists(p)]
    if missing:
        print("ERROR: corpus is missing:")
        for p in missing:
            print("  -", p)
        print("\nCopy the S20 corpus first, e.g.:")
        print("  cp -r ../../Session_20/Code/data .")
        return 1

    print(f"\nBenchmark settings:  K={args.k}  ·  strategies={args.strategies}")
    print(f"Corpus:  {[os.path.basename(p) for p in DEFAULT_CORPUS]}")
    print(f"Test queries:  {len(TEST_QUERIES)}")

    docs = load_corpus(DEFAULT_CORPUS)

    results: List[StrategyResult] = []
    for name in args.strategies:
        if name not in STRATEGIES:
            print(f"  WARN: unknown strategy {name!r} — skipped")
            continue
        try:
            r = evaluate_strategy(name, STRATEGIES[name], docs, TEST_QUERIES, args.k)
            results.append(r)
        except Exception as e:
            print(f"  ERROR running strategy {name!r}: {e}")

    if results:
        print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
