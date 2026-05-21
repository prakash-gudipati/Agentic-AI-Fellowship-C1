"""
Session 26 — demo.py

The signature demo for Observability — Tracing, Debugging + Cost.

Takes the S25 RAG-with-evals pipeline and adds the S26 observability
layer WITHOUT rewriting any pipeline code:

  * LangSmith tracing via @track decorators (graceful offline fallback)
  * Auto-instrumented LLM clients via wrap_llm_client() — token counts
    and dollar cost come through automatically
  * Cost budget gate enforcement via observability.cost_budget (pulls
    actual run cost from the LangSmith API, falls back to the offline
    JSONL log when LangSmith isn't reachable)
  * Trace-driven debugging via observability.debugger.replay — open a
    failed run, replay it with a different model, compare side-by-side

The demo runs in five acts (down from seven in the Opik-anchored draft
— `cost_breakdown` and `dashboard` were removed because LangSmith
already provides both in its hosted UI; we point students there
instead of duplicating the work in terminal code):

  ACT 1  - cold-cache build (traced)
  ACT 2  - warm-cache rebuild (traced; cache fully populated)
  ACT 3  - golden-dataset eval — every row produces a LangSmith trace
  ACT 4  - find the lowest-scoring row, REPLAY with a stronger model,
           print side-by-side diff
  ACT 5  - cost budget gate — pass + fail demos (LangSmith if active,
           offline JSONL otherwise)

Usage:
    python demo.py
    python demo.py --skip-evals          # ACT 1, 2 only — fast smoke
    python demo.py --queries-per-day 5000
    python demo.py --daily-budget 2.50
    python demo.py --replay-model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from loaders import load_corpus
from chunker import chunk_corpus
from embedding_cache import CachedChunkIndex, EmbeddingCache
from retrievers.similarity import SimilarityRetriever
from pipeline import RAGPipeline
from evals.golden_dataset import load_golden_dataset
from evals import harness_ragas
from observability.langsmith_tracing import (is_langsmith_active,
                                              get_offline_tracer,
                                              get_project_url)
from observability.cost_budget import (check_budget,
                                        check_budget_synthetic,
                                        BudgetExceededError)
from observability.debugger import replay


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

def act_1_cold_cache(cache: EmbeddingCache) -> CachedChunkIndex:
    banner("ACT 1  -  cold-cache build (TRACED)")
    docs = load_corpus(DEFAULT_CORPUS)
    chunks = chunk_corpus(docs)
    t0 = time.time()
    index = CachedChunkIndex(cache=cache)
    index.add(chunks)
    print(f"\n[act1] built in {time.time() - t0:.2f}s  "
          f"cache stats: {cache.stats()}")
    return index


# ── ACT 2 ───────────────────────────────────────────────────────────────────

def act_2_warm_cache(cache: EmbeddingCache) -> None:
    banner("ACT 2  -  warm-cache rebuild (TRACED, same texts -> all hits)")
    docs = load_corpus(DEFAULT_CORPUS)
    chunks = chunk_corpus(docs)
    t0 = time.time()
    index2 = CachedChunkIndex(cache=cache)
    index2.add(chunks)
    print(f"\n[act2] built in {time.time() - t0:.2f}s  "
          f"cache stats: {cache.stats()}")


# ── ACT 3 ───────────────────────────────────────────────────────────────────

def act_3_traced_eval(pipeline: RAGPipeline) -> Dict[str, Any]:
    """Run Ragas eval — every row is now a traced pipeline call."""
    banner("ACT 3  -  RAGAS DEEP eval, EVERY ROW TRACED")
    rows = load_golden_dataset()
    summary = harness_ragas.evaluate(
        pipeline, rows,
        show_one_judge_prompt=False,
        verbose=False,
    )
    print()
    print(f"  faithfulness:       {summary['faithfulness']:.3f}")
    print(f"  answer_relevancy:   {summary['answer_relevancy']:.3f}")
    print(f"  context_precision:  {summary['context_precision']:.3f}")
    print(f"  context_recall:     {summary['context_recall']:.3f}")
    print(f"  judge LLM calls:    {summary['judge_calls']}")
    url = get_project_url()
    if url:
        print(f"\n  [LangSmith] open the project to inspect traces:")
        print(f"   {url}")
    else:
        offline = get_offline_tracer().trace_file
        print(f"\n  [offline] traces appended to: {offline}")
    return summary


# ── ACT 4 ───────────────────────────────────────────────────────────────────

def act_4_debug_via_trace(pipeline: RAGPipeline,
                          summary: Dict[str, Any],
                          replay_model: str) -> None:
    """Open the lowest-faithfulness row's run and REPLAY with a stronger
    model side-by-side."""
    banner("ACT 4  -  TRACE-DRIVEN DEBUGGING  -  replay with stronger model")
    rows = summary.get("rows", []) or []
    if not rows:
        print("[act4] no per-row scores available; skipping replay")
        return
    worst = min(rows,
                 key=lambda r: float(r.get("faithfulness", 1.0) or 1.0))
    question = worst.get("question", "")
    score = worst.get("faithfulness", 0.0)
    print(f"[act4] worst row:  faithfulness={float(score):.2f}  q={question!r}")

    # Find the run_id of the eval call for this question. The offline
    # tracer stores the question inside the root run's inputs.kwargs.
    original_run_id: Optional[str] = None
    for r in get_offline_tracer().get_recent(n=50, only_roots=True):
        inputs = r.get("inputs") or {}
        kwargs = inputs.get("kwargs") or {} if isinstance(inputs, dict) else {}
        args = inputs.get("args") or [] if isinstance(inputs, dict) else []
        candidate = kwargs.get("question") if isinstance(kwargs, dict) else None
        if candidate is None and args:
            candidate = args[0] if args else None
        if isinstance(candidate, str) and candidate.strip() == question.strip():
            original_run_id = r.get("run_id")
            break
    if original_run_id is None:
        print("[act4] couldn't locate the run for that question — "
              "replay will run fresh, no original-vs-replay diff")

    def factory(model: str = replay_model, **kw):
        # Build a fresh pipeline that shares the original retriever (no
        # re-indexing cost), but swap in the stronger model.
        return RAGPipeline(pipeline.retriever, k=pipeline.k, model=model)

    replay(factory,
           run_id=original_run_id,
           question=question,
           knob="model", new_value=replay_model,
           verbose=True)


# ── ACT 5 ───────────────────────────────────────────────────────────────────

def act_5_budget_gate(*,
                       queries_per_day: int,
                       daily_budget_usd: float) -> None:
    """Two scenarios: pass on a generous budget, fail on a tight one.

    Both scenarios pull cost data from the SAME source — LangSmith if
    active, the offline JSONL log otherwise. The third (synthetic) call
    fakes the numbers so the fail tier triggers without burning tokens.
    """
    banner("ACT 5  -  COST BUDGET GATE  -  pass + fail demos")

    print("\n[act5a] PASS scenario — generous budget vs the dev sample:")
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

    print("\n[act5b] FAIL scenario — tight budget gates an 'expensive' run:")
    try:
        check_budget_synthetic(sample_cost_usd=0.50, sample_queries=10,
                                queries_per_day=queries_per_day,
                                daily_budget_usd=daily_budget_usd,
                                raise_on_fail=True)
    except BudgetExceededError as e:
        print(f"\n[act5b]  GATE FAILED (expected):  {e}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="S26 observability demo")
    parser.add_argument("--skip-evals", action="store_true",
                        help="skip ACTs 3-5 (fast smoke: act 1, 2 only)")
    parser.add_argument("--replay-model", type=str,
                        default="claude-sonnet-4-6",
                        help="stronger model for the act-4 trace replay")
    parser.add_argument("--queries-per-day", type=int, default=10_000,
                        help="projected production traffic for the budget gate")
    parser.add_argument("--daily-budget", type=float, default=5.00,
                        help="$USD daily budget threshold")
    args = parser.parse_args()

    print(f"[demo] LANGSMITH_ACTIVE = {is_langsmith_active()}")
    if is_langsmith_active():
        print(f"[demo] LangSmith project: {get_project_url()}")
    else:
        print(f"[demo] offline traces will be appended to "
              f"{get_offline_tracer().trace_file}")

    missing = [p for p in DEFAULT_CORPUS if not os.path.exists(p)]
    if missing:
        print("ERROR: corpus files are missing:")
        for p in missing:
            print("  -", p)
        return 1

    cache = EmbeddingCache()
    index = act_1_cold_cache(cache)
    act_2_warm_cache(cache)

    if not args.skip_evals:
        retriever = SimilarityRetriever(index)
        pipeline = RAGPipeline(retriever, k=4)
        summary = act_3_traced_eval(pipeline)
        act_4_debug_via_trace(pipeline, summary,
                               replay_model=args.replay_model)
        act_5_budget_gate(queries_per_day=args.queries_per_day,
                           daily_budget_usd=args.daily_budget)
    else:
        banner("ACTs 3-5 SKIPPED  (--skip-evals)")

    banner("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
