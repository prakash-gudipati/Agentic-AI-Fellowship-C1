"""
Session 25 — demo.py

The signature demo. Takes the S24-era RAG pipeline and stacks today's
two production additions onto it:

  1. embedding cache       (content-hash cache so unchanged chunks
                            never re-embed)
  2. evaluation framework  (Ragas DEEP with all 5 metrics —
                            faithfulness, answer_relevancy,
                            context_precision, context_recall,
                            AspectCritic)

Plus two diagnostic moves:
  3. score-shape -> recommended fix (diagnostic.py)
  4. synthetic test-data preview (TestsetGenerator)

DeepEval is deferred to S41 (Guardrails + Evals) and S42 (Eval-Driven
Development) where its safety + workflow features land naturally.

The demo runs in six acts.

Usage:
    python demo.py
    python demo.py --skip-evals
    python demo.py --skip-synthetic
    python demo.py --notes "..."
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from loaders import load_corpus
from chunker import chunk_corpus
from embedding_cache import CachedChunkIndex, EmbeddingCache
from retrievers.similarity import SimilarityRetriever
from pipeline import RAGPipeline
from evals.golden_dataset import load_golden_dataset
from evals import harness_custom, harness_ragas
from evals.synthetic_dataset import generate_synthetic_rows
from evals.diagnostic import read_scores, print_report
from evals.score_log import append_run


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


def act_1_cold_cache(cache: EmbeddingCache) -> CachedChunkIndex:
    """ACT 1 - build the index with a cold cache."""
    banner("ACT 1  -  cold-cache build (every chunk is a miss)")
    docs = load_corpus(DEFAULT_CORPUS)
    chunks = chunk_corpus(docs)
    t0 = time.time()
    index = CachedChunkIndex(cache=cache)
    index.add(chunks)
    print(f"\n[act1] built in {time.time() - t0:.2f}s  "
          f"cache stats: {cache.stats()}")
    return index


def act_2_warm_cache(cache: EmbeddingCache) -> None:
    """ACT 2 - rebuild the same index, should now hit cache."""
    banner("ACT 2  -  warm-cache rebuild (same texts -> all hits)")
    docs = load_corpus(DEFAULT_CORPUS)
    chunks = chunk_corpus(docs)
    t0 = time.time()
    index2 = CachedChunkIndex(cache=cache)
    index2.add(chunks)
    print(f"\n[act2] built in {time.time() - t0:.2f}s  "
          f"cache stats: {cache.stats()}")


def act_3_ragas_deep(pipeline: RAGPipeline) -> Dict[str, Any]:
    """ACT 3 - Ragas: all 5 metrics, visible judge prompt, cost summary."""
    banner("ACT 3  -  RAGAS DEEP  (5 metrics incl. AspectCritic)")
    rows = load_golden_dataset()
    summary = harness_ragas.evaluate(
        pipeline, rows,
        aspect_critic_definition=(
            "Does the answer cite a source from the retrieved context "
            "(document name, 'according to', 'per the manual', or a "
            "direct quote)?"
        ),
        show_one_judge_prompt=True,
        verbose=False,
    )
    print()
    print(f"  framework:                  {summary['framework']}")
    print(f"  judge_model:                {summary['judge_model']}")
    print(f"  faithfulness:               {summary['faithfulness']:.3f}")
    print(f"  answer_relevancy:           {summary['answer_relevancy']:.3f}")
    print(f"  context_precision:          {summary['context_precision']:.3f}")
    print(f"  context_recall:             {summary['context_recall']:.3f}")
    print(f"  aspect_critic (cites_src):  {summary['aspect_critic']:.3f}")
    print(f"  judge LLM calls this run:   {summary['judge_calls']}")
    return summary


def act_4_diagnostic(ragas_summary: Dict[str, Any]) -> None:
    """ACT 4 - read the scores into a recommended fix."""
    banner("ACT 4  -  DIAGNOSTIC READING  -  score shape  ->  layer to fix")
    findings = read_scores(ragas_summary)
    print_report(findings)


def act_5_synthetic_preview() -> None:
    """ACT 5 - preview Ragas TestsetGenerator (or offline fallback)."""
    banner("ACT 5  -  SYNTHETIC TEST-DATA PREVIEW  (TestsetGenerator)")
    docs = load_corpus(DEFAULT_CORPUS)
    rows = generate_synthetic_rows(docs, n_rows=3)
    if not rows:
        print("[act5] generator produced no rows - check API key / packages")
        return
    for i, r in enumerate(rows, start=1):
        print(f"\n  --- synthetic row {i} ---")
        print(f"  Q:  {r.question}")
        print(f"  A:  {r.ground_truth[:120]}"
              f"{'...' if len(r.ground_truth) > 120 else ''}")
        if r.ground_truth_contexts:
            ev = r.ground_truth_contexts[0]
            print(f"  C:  {ev[:120]}{'...' if len(ev) > 120 else ''}")


def act_6_score_log(rs: Dict[str, Any], cache: EmbeddingCache,
                    notes: str) -> None:
    """ACT 6 - append today's run to the regression CSV."""
    banner("ACT 6  -  APPEND TO score_log.csv  (the regression log)")
    append_run({
        "retriever_name":     "similarity",
        "cache_hit_rate":     cache.stats()["hit_rate"],
        "faithfulness":       rs["faithfulness"],
        "answer_relevancy":   rs["answer_relevancy"],
        "context_precision":  rs["context_precision"],
        "context_recall":     rs["context_recall"],
        "notes":              notes or "demo.py - Ragas deep dive",
    })


def main() -> int:
    parser = argparse.ArgumentParser(description="S25 Ragas demo")
    parser.add_argument("--skip-evals", action="store_true",
                        help="skip ACTs 3, 4, 6 (faster smoke test)")
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="skip ACT 5 (no LLM bootstrap)")
    parser.add_argument("--notes", type=str, default="",
                        help="free-form text added to the score-log row")
    args = parser.parse_args()

    missing = [p for p in DEFAULT_CORPUS if not os.path.exists(p)]
    if missing:
        print("ERROR: corpus files are missing:")
        for p in missing:
            print("  -", p)
        return 1

    cache = EmbeddingCache()
    index = act_1_cold_cache(cache)
    act_2_warm_cache(cache)

    if args.skip_evals:
        banner("ACTs 3, 4, 6 SKIPPED  (--skip-evals)")
    else:
        retriever = SimilarityRetriever(index)
        pipeline = RAGPipeline(retriever, k=4)

        ragas_summary = act_3_ragas_deep(pipeline)
        act_4_diagnostic(ragas_summary)
        act_6_score_log(ragas_summary, cache, args.notes)

    if not args.skip_synthetic:
        act_5_synthetic_preview()
    else:
        banner("ACT 5 SKIPPED  (--skip-synthetic)")

    banner("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
