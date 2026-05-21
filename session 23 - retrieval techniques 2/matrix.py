"""
Session 23 — matrix.py

THE SIGNATURE DEMO. The 25-combination matrix.

5 chunkers (rows)  ×  5 retrievers (columns)  =  25 cells.
Every cell runs the same fixed query set through that chunker+retriever
combo and reports a single score: hit-rate at K.

Why this matters: in production RAG, the SINGLE biggest decision is which
(chunker, retriever) combination to use. There is no universal winner.
The combo that wins for FAQ docs is wrong for long manuals. Measure on
YOUR corpus before you commit.

Usage:
    python matrix.py                  # default K=3, all 25 cells
    python matrix.py --k 5            # top-5 retrieval
    python matrix.py --skip-rerank    # skip the LLM column (no API key)
    python matrix.py --skip-semantic  # skip the semantic row (no embed calls)

Notes on scoring:
  Hit-rate = fraction of test queries where the expected source appears in
  the top-K results. Simple, transparent, comparable across cells.
  Production teams use richer metrics (MRR, nDCG, Ragas faithfulness) — see
  S25.

Production patterns shown:
  - retriever strategy pattern (S22)         — every retriever swappable
  - chunker strategy pattern (S21)           — every chunker swappable
  - SCORE NORMALISATION BEFORE FUSION (S23)  — see HybridRetriever
  - TWO-STAGE RETRIEVAL (S23)                — see LLMRerankRetriever
  - MEASURE BEFORE YOU GUESS (S23)           — this script IS the pattern
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

# Make sibling modules importable when running as a script.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from loaders import load_corpus
from chunkers import CHUNKERS
from retrievers.base import ChunkIndex
from retrievers.similarity import SimilarityRetriever
from retrievers.bm25 import BM25Retriever
from retrievers.hybrid import HybridRetriever
from retrievers.mmr import MMRRetriever
from retrievers.rerank import LLMRerankRetriever
from test_queries import TEST_QUERIES


DEFAULT_CORPUS = [
    os.path.join(HERE, "data", "intro_to_rag.pdf"),
    os.path.join(HERE, "data", "product_manual.pdf"),
    os.path.join(HERE, "data", "sample_article.html"),
]

# Order matters — this is the column order in the printed table.
RETRIEVER_ORDER = ["similarity", "bm25", "hybrid", "mmr", "rerank"]


def build_retriever(name: str, index: ChunkIndex):
    """Factory — defaults chosen to match what the slides describe."""
    if name == "similarity":
        return SimilarityRetriever(index)
    if name == "bm25":
        return BM25Retriever(index)
    if name == "hybrid":
        return HybridRetriever(index, mode="weighted", alpha=0.5)
    if name == "mmr":
        return MMRRetriever(index)
    if name == "rerank":
        return LLMRerankRetriever(index,
                                  base=SimilarityRetriever(index),
                                  fetch_k=10)
    raise ValueError(f"unknown retriever {name!r}")


@dataclass
class CellResult:
    chunker: str
    retriever: str
    hits: int
    total: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


def run_cell(chunker_name: str, retriever_name: str,
             docs: list, k: int) -> CellResult:
    """Build the chunker+retriever combo, run every test query, count hits."""
    print(f"\n── CELL  chunker={chunker_name}  retriever={retriever_name} ──")
    chunks = CHUNKERS[chunker_name](docs)
    index = ChunkIndex()
    index.add(chunks)

    retriever = build_retriever(retriever_name, index)

    hits = 0
    for q in TEST_QUERIES:
        try:
            results = retriever.pick(q.question, k=k)
        except Exception as e:
            print(f"[matrix] WARN: query failed in cell "
                  f"({chunker_name},{retriever_name}): {e!s}")
            continue
        sources = [r.chunk.source for r in results]
        if q.expected_source in sources:
            hits += 1

    return CellResult(chunker=chunker_name, retriever=retriever_name,
                      hits=hits, total=len(TEST_QUERIES))


def print_matrix(results: List[CellResult],
                 chunker_names: List[str],
                 retriever_names: List[str]) -> None:
    """Pretty-print the matrix as a hit-rate table."""
    # Index results for fast lookup.
    lookup: Dict[tuple, CellResult] = {
        (r.chunker, r.retriever): r for r in results
    }

    col_w = 11
    name_w = 12

    print()
    print("═" * 78)
    print("  HIT-RATE MATRIX   (fraction of queries where expected source is in top-K)")
    print("═" * 78)
    # Header
    header = " " * name_w + "".join(f"{rn:>{col_w}}" for rn in retriever_names)
    print(header)
    print("  " + "-" * (len(header) - 2))

    best_cell = None
    best_rate = -1.0
    for cn in chunker_names:
        row = f"  {cn:<{name_w - 2}}"
        for rn in retriever_names:
            cell = lookup.get((cn, rn))
            if cell is None:
                row += f"{'—':>{col_w}}"
            else:
                pct = cell.hit_rate * 100
                row += f"{cell.hits}/{cell.total} ({pct:>3.0f}%)".rjust(col_w)
                if cell.hit_rate > best_rate:
                    best_rate = cell.hit_rate
                    best_cell = cell
        print(row)
    print()
    if best_cell:
        print(f"  BEST CELL  →  chunker={best_cell.chunker!r}  "
              f"retriever={best_cell.retriever!r}  "
              f"hit-rate={best_cell.hit_rate * 100:.0f}%")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="S23 5×5 matrix")
    parser.add_argument("--k", type=int, default=3,
                        help="top-K for retrieval (default 3)")
    parser.add_argument("--skip-rerank", action="store_true",
                        help="skip the rerank column (no LLM calls)")
    parser.add_argument("--skip-semantic", action="store_true",
                        help="skip the semantic row (no embed calls for chunking)")
    args = parser.parse_args()

    missing = [p for p in DEFAULT_CORPUS if not os.path.exists(p)]
    if missing:
        print("ERROR: corpus is missing:")
        for p in missing:
            print("  -", p)
        return 1

    print("\n" + "═" * 78)
    print("INDEX-TIME — loading corpus (chunkers run per-cell)")
    print("═" * 78)
    docs = load_corpus(DEFAULT_CORPUS)

    # Build the run plan.
    chunker_names = [name for name in CHUNKERS
                     if not (args.skip_semantic and name == "semantic")]
    retriever_names = [name for name in RETRIEVER_ORDER
                       if not (args.skip_rerank and name == "rerank")]

    n_cells = len(chunker_names) * len(retriever_names)
    print(f"\nRunning {n_cells} cells  "
          f"({len(chunker_names)} chunkers × {len(retriever_names)} retrievers)  "
          f"K={args.k}  queries={len(TEST_QUERIES)}")

    results: List[CellResult] = []
    for cn in chunker_names:
        for rn in retriever_names:
            results.append(run_cell(cn, rn, docs, k=args.k))

    print_matrix(results, chunker_names, retriever_names)

    print("  Reading the matrix:")
    print("   - Hit-rate = (queries where expected source is in top-K) / total")
    print("   - The grid is a benchmarking WORKFLOW, not a one-time exercise.")
    print("   - The winner for THIS corpus + THESE queries is not the winner")
    print("     for yours. Re-run with your domain documents and queries.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
