"""
Session 27 -- demo.py

THE CAPSTONE WALKTHROUGH SCRIPT.

This is what the instructor narrates over on screen-share. No live
typing, no exercise. The whole point is: the student SEES every
S20-S26 piece compose into one production-grade pipeline, then leaves
with `Session_27_Capstone_Brief.docx` to do it themselves on their
own corpus.

THE FIVE ACTS
-------------
  ACT 1  -  index the corpus into ChromaDB (or fall back to in-memory)
  ACT 2  -  single traced query through the baseline retriever
  ACT 3  -  COMPARISON HARNESS -- three retrievers, ten golden rows,
            one ranked table
  ACT 4  -  the defended pick -- print the README stub that would
            justify shipping the winner
  ACT 5  -  budget gate -- pass + fail demos against the LangSmith
            (or offline) cost ledger

Run:
    python demo.py
    python demo.py --skip-eval           # acts 1+2 only -- smoke
    python demo.py --skip-comparison     # acts 1+2+4+5 -- baseline only
    python demo.py --persist-dir ./my-chroma --collection my-capstone
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from loaders import load_corpus
from chunker import chunk_corpus
from embedding_cache import CachedChunkIndex, EmbeddingCache
from retrievers import (SimilarityRetriever, MMRRetriever, HybridRetriever,
                        ChunkIndex)
from pipeline import RAGPipeline
from evals.golden_dataset import load_golden_dataset
from evals.comparison_harness import (compare_retrievers, RetrieverConfig,
                                       ComparisonResult)
from observability.langsmith_tracing import (is_langsmith_active,
                                              get_offline_tracer,
                                              get_project_url)
from observability.cost_budget import (check_budget, check_budget_synthetic,
                                        BudgetExceededError)

# ChromaDB is the production default; the demo falls back gracefully.
from chroma_index import ChromaChunkIndex, is_chromadb_available


DEFAULT_CORPUS = [
    os.path.join(HERE, "data", "intro_to_rag.pdf"),
    os.path.join(HERE, "data", "product_manual.pdf"),
    os.path.join(HERE, "data", "sample_article.html"),
]


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ── ACT 1 ───────────────────────────────────────────────────────────────────

def act_1_index(corpus_files: List[str],
                 *,
                 collection: str,
                 persist_dir: str):
    """Build (or rehydrate) the index. ChromaDB if available, else
    in-memory."""
    banner("ACT 1  -  index the corpus")

    docs = load_corpus(corpus_files)
    chunks = chunk_corpus(docs)
    print(f"\n[act1] loaded {len(docs)} docs -> {len(chunks)} chunks")

    if is_chromadb_available():
        print(f"[act1] backend: ChromaDB (persist_dir={persist_dir!r}, "
              f"collection={collection!r})")
        index = ChromaChunkIndex(collection=collection,
                                  persist_dir=persist_dir)
    else:
        print(f"[act1] backend: in-memory fallback "
              f"(chromadb not installed; the capstone REQUIRES it)")
        index = ChunkIndex()

    t0 = time.time()
    index.add(chunks)
    print(f"[act1] indexed in {time.time() - t0:.2f}s "
          f"(len={len(index)})")
    return index


# ── ACT 2 ───────────────────────────────────────────────────────────────────

def act_2_baseline_query(index, query: str = "What is RAG?") -> None:
    """One traced query through the SimilarityRetriever baseline."""
    banner("ACT 2  -  single traced query (baseline = similarity)")
    pipeline = RAGPipeline(SimilarityRetriever(index), k=4)
    result = pipeline.answer(query)
    print("\n--- ANSWER ---")
    print(result.answer)
    print("--------------")
    print(f"cost: ${result.cost_usd:.6f}  "
          f"in={result.input_tokens} out={result.output_tokens}")

    if is_langsmith_active():
        url = get_project_url()
        if url:
            print(f"\n[act2] open the trace in LangSmith:\n  {url}")
    else:
        tr_file = get_offline_tracer().trace_file
        print(f"\n[act2] offline traces at: {tr_file}")


# ── ACT 3 ───────────────────────────────────────────────────────────────────

def act_3_comparison(index) -> ComparisonResult:
    """The capstone's signature artifact -- three retrievers ranked."""
    banner("ACT 3  -  COMPARISON HARNESS  -  three retrievers ranked")

    def factory_similarity():
        return RAGPipeline(SimilarityRetriever(index), k=4)

    def factory_mmr():
        return RAGPipeline(
            MMRRetriever(index, fetch_k=12, lambda_=0.6), k=4)

    def factory_hybrid():
        return RAGPipeline(HybridRetriever(index, alpha=0.5), k=4)

    configs = [
        RetrieverConfig(name="similarity",
                        factory=factory_similarity,
                        notes="S22 baseline -- pure top-K cosine"),
        RetrieverConfig(name="mmr-k4-lambda0.6",
                        factory=factory_mmr,
                        notes="S22 -- diversity-aware re-rank"),
        RetrieverConfig(name="hybrid-alpha0.5",
                        factory=factory_hybrid,
                        notes="S23 -- 50/50 dense + BM25"),
    ]

    rows = load_golden_dataset()
    return compare_retrievers(configs, rows, verbose=True)


# ── ACT 4 ───────────────────────────────────────────────────────────────────

def act_4_defended_pick(comparison: ComparisonResult,
                         corpus_files: List[str]) -> None:
    """Print the README stub the student would commit alongside the
    capstone pick. THIS IS THE DELIVERABLE the rubric grades."""
    banner("ACT 4  -  THE DEFENDED PICK  -  README stub")
    if not comparison.rows:
        print("[act4] comparison produced no rows; nothing to defend.")
        return

    ordered = sorted(comparison.rows, key=lambda r: -r.composite)
    winner = ordered[0]
    print()
    print("------ README.md (defence stub) ------")
    print()
    print(f"# Capstone -- Defended Pick")
    print()
    print(f"**Shipped retriever:** `{winner.name}`")
    print(f"**Notes:** {winner.notes or '(none)'}")
    print()
    print("## Why this retriever won")
    print()
    print("I compared three retrieval strategies against the same")
    print(f"{comparison.golden_n}-row golden set:")
    print()
    print("| Rank | Retriever | Composite | Faithfulness | Recall |")
    print("|------|-----------|-----------|--------------|--------|")
    for i, r in enumerate(ordered, start=1):
        print(f"| {i} | `{r.name}` | {r.composite:.3f} | "
              f"{r.faithfulness:.3f} | {r.context_recall:.3f} |")
    print()
    print(f"Composite weights: `{comparison.weights}`.")
    print()
    print("## Corpus")
    print()
    for f in corpus_files:
        print(f"- `{os.path.basename(f)}`")
    print()
    print("## Reproducing this comparison")
    print()
    print("```bash")
    print("python demo.py                                     # full run")
    print("python demo.py --skip-comparison                   # baseline only")
    print("```")
    print()
    print("--------------------------------------")


# ── ACT 5 ───────────────────────────────────────────────────────────────────

def act_5_budget_gate(*,
                       queries_per_day: int,
                       daily_budget_usd: float) -> None:
    banner("ACT 5  -  COST BUDGET GATE  -  pass + fail demos")

    print("\n[act5a] PASS scenario -- generous budget vs the dev sample:")
    try:
        check_budget(last_n=10,
                      queries_per_day=queries_per_day,
                      daily_budget_usd=daily_budget_usd,
                      raise_on_fail=False)
    except RuntimeError as e:
        print(f"[act5a] no traced runs yet ({e!s}); using synthetic numbers")
        check_budget_synthetic(sample_cost_usd=0.012, sample_queries=10,
                                queries_per_day=queries_per_day,
                                daily_budget_usd=daily_budget_usd,
                                raise_on_fail=False)

    print("\n[act5b] FAIL scenario -- tight budget gates an "
          "'expensive' run:")
    try:
        check_budget_synthetic(sample_cost_usd=0.50, sample_queries=10,
                                queries_per_day=queries_per_day,
                                daily_budget_usd=daily_budget_usd,
                                raise_on_fail=True)
    except BudgetExceededError as e:
        print(f"\n[act5b]  GATE FAILED (expected):  {e}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="S27 RAG Capstone walkthrough demo")
    parser.add_argument("--skip-eval", action="store_true",
                        help="skip ACTs 3-5 (fast smoke)")
    parser.add_argument("--skip-comparison", action="store_true",
                        help="skip ACT 3 (no comparison; baseline only)")
    parser.add_argument("--persist-dir", type=str, default=".chroma",
                        help="ChromaDB persist directory")
    parser.add_argument("--collection", type=str,
                        default="fellowship-capstone",
                        help="ChromaDB collection name")
    parser.add_argument("--queries-per-day", type=int, default=10_000,
                        help="projected production traffic for budget gate")
    parser.add_argument("--daily-budget", type=float, default=5.00,
                        help="$USD daily budget threshold")
    args = parser.parse_args()

    print(f"[demo] LANGSMITH_ACTIVE   = {is_langsmith_active()}")
    print(f"[demo] CHROMADB_AVAILABLE = {is_chromadb_available()}")

    missing = [p for p in DEFAULT_CORPUS if not os.path.exists(p)]
    if missing:
        print("ERROR: corpus files are missing:")
        for p in missing:
            print("  -", p)
        return 1

    index = act_1_index(DEFAULT_CORPUS,
                         collection=args.collection,
                         persist_dir=args.persist_dir)
    act_2_baseline_query(index)

    if args.skip_eval:
        banner("ACTs 3-5 SKIPPED  (--skip-eval)")
        return 0

    comparison = None
    if not args.skip_comparison:
        comparison = act_3_comparison(index)
        act_4_defended_pick(comparison, DEFAULT_CORPUS)
    else:
        banner("ACT 3 SKIPPED  (--skip-comparison)")

    act_5_budget_gate(queries_per_day=args.queries_per_day,
                       daily_budget_usd=args.daily_budget)

    banner("DONE  -  capstone walkthrough complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
