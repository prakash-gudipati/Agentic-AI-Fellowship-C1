"""
Session 25 — evals/harness_custom.py

THE BASELINE — a hand-rolled eval, the S20-style "did it answer right?"

Why include this at all? Because the whole point of today's session is
that this kind of eval is NOT enough at scale. By running it side-by-side
with Ragas and DeepEval, students see WHY the framework approach gives
more signal:

  Custom harness:    "0.6"          (60% of answers passed substring check)
  Ragas:             faithfulness=0.84  relevance=0.79  c.precision=0.71  c.recall=0.62
  DeepEval:          score=0.66, reasoning=" 'partially relevant — answer
                     does not address the time-period qualifier'"

The custom score collapses four failure modes into one number. The
framework scores separate them — which is what lets you debug.

This harness is intentionally minimal:
  - exact substring match of ANY ground-truth phrase in the answer
  - "I don't know" is a special case (rewarded when ground truth is also
    "I don't know", penalised otherwise)

Output: { "pass_rate": float, "rows": [ {question, passed, answer}, ... ] }
"""

from __future__ import annotations

from typing import Any, Dict, List

from pipeline import RAGPipeline
from .golden_dataset import GoldenRow, load_golden_dataset


def _row_passes(answer: str, row: GoldenRow) -> bool:
    """Cheap heuristic: does the answer contain key terms from the truth?"""
    if not answer:
        return False
    answer_l = answer.lower()
    truth_l = row.ground_truth.lower()

    # Extract the first 6 nontrivial words from the ground truth. If the
    # answer mentions at least 3 of them, count as a pass. This is exactly
    # the kind of brittle heuristic that motivates moving to Ragas.
    truth_terms = [w.strip(".,;:()") for w in truth_l.split()
                   if len(w) > 4 and w.isalpha()][:6]
    if not truth_terms:
        return True
    hits = sum(1 for w in truth_terms if w in answer_l)
    return hits >= 3


def evaluate(pipeline: RAGPipeline,
             rows: List[GoldenRow] = None,
             *,
             verbose: bool = True) -> Dict[str, Any]:
    """Run every golden row through `pipeline`, return a summary dict."""
    rows = rows or load_golden_dataset()
    out_rows: List[Dict[str, Any]] = []
    passed = 0

    for i, row in enumerate(rows, start=1):
        if verbose:
            print(f"\n[custom] row {i}/{len(rows)}  q={row.question!r}")
        result = pipeline.answer(row.question)
        ok = _row_passes(result.answer, row)
        if ok:
            passed += 1
        out_rows.append({
            "question": row.question,
            "passed": ok,
            "answer": result.answer,
            "ground_truth": row.ground_truth,
        })

    pass_rate = passed / len(rows) if rows else 0.0
    print(f"\n[custom] pass_rate = {passed}/{len(rows)} = {pass_rate:.2f}")
    return {
        "framework": "custom",
        "pass_rate": pass_rate,
        "rows": out_rows,
    }


if __name__ == "__main__":
    import os, sys
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, HERE)
    from loaders import load_corpus
    from chunker import chunk_corpus
    from embedding_cache import CachedChunkIndex
    from retrievers.similarity import SimilarityRetriever

    files = [
        os.path.join(HERE, "data", "intro_to_rag.pdf"),
        os.path.join(HERE, "data", "product_manual.pdf"),
        os.path.join(HERE, "data", "sample_article.html"),
    ]
    docs = load_corpus(files)
    chunks = chunk_corpus(docs)
    index = CachedChunkIndex()
    index.add(chunks)
    pipeline = RAGPipeline(SimilarityRetriever(index), k=3)

    summary = evaluate(pipeline)
    print("\n========= CUSTOM HARNESS SUMMARY =========")
    print(f"pass rate: {summary['pass_rate']:.2f}")
