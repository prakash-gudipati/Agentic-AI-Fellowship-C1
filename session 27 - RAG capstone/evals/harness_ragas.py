"""
Session 25 — evals/harness_ragas.py

THE FRAMEWORK — Ragas. Production-grade RAG eval.

Today we cover FIVE Ragas metrics:
  faithfulness, answer_relevancy, context_precision, context_recall,
  and AspectCritic (the escape-hatch for ANY custom rubric phrased in English).

JUDGE-MODEL CONFIGURATION
-------------------------
Every Ragas metric is an LLM-as-judge call. We expose judge_model so
the score is reproducible across teammates.

COST MODEL
----------
For N rows × M metrics, expect roughly N × (2+1+1+1+1) judge calls.
For 10 rows × 5 metrics that's ~60 calls. We print the running call
count so the bill is visible.

OFFLINE FALLBACK
----------------
If the `ragas` package is not installed, this harness falls back to
an in-house LLM-as-judge implementation using the Anthropic SDK.
Same metric shapes, slightly different score values.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from pipeline import RAGPipeline
from .golden_dataset import GoldenRow, load_golden_dataset


load_dotenv()


DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"


def evaluate(pipeline: RAGPipeline,
             rows: Optional[List[GoldenRow]] = None,
             *,
             judge_model: str = DEFAULT_JUDGE_MODEL,
             aspect_critic_definition: Optional[str] = None,
             show_one_judge_prompt: bool = True,
             verbose: bool = True) -> Dict[str, Any]:
    """Run Ragas (or the offline fallback) over the golden set."""
    rows = rows or load_golden_dataset()
    aspect_definition = aspect_critic_definition or (
        "Does the answer either cite a source from the retrieved context "
        "or use phrases like 'according to the document' or 'per the manual'? "
        "Pass if there is any source attribution."
    )

    print(f"\n[ragas] judge_model={judge_model}")
    print(f"[ragas] running pipeline against {len(rows)} golden rows...")

    run_outputs: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        if verbose:
            print(f"[ragas] pipeline row {i}/{len(rows)} q={row.question!r}")
        res = pipeline.answer(row.question)
        run_outputs.append({
            "question": row.question,
            "answer": res.answer,
            "contexts": res.contexts,
            "ground_truth": row.ground_truth,
        })

    try:
        return _evaluate_with_ragas(run_outputs,
                                    judge_model=judge_model,
                                    aspect_definition=aspect_definition)
    except ImportError:
        print("[ragas] WARN: 'ragas' package not installed — using "
              "the S25 offline fallback (LLM-as-judge in pure Anthropic SDK).")
        return _evaluate_with_fallback(
            run_outputs, judge_model=judge_model,
            aspect_definition=aspect_definition,
            show_one_judge_prompt=show_one_judge_prompt)


def _evaluate_with_ragas(run_outputs: List[Dict[str, Any]], *,
                         judge_model: str, aspect_definition: str
                         ) -> Dict[str, Any]:
    """Use the real ragas package with all five metrics."""
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (faithfulness, answer_relevancy,
                                context_precision, context_recall)
    AspectCriticMetric: Any
    try:
        from ragas.metrics import AspectCritic as AspectCriticMetric
    except ImportError:
        try:
            from ragas.metrics import AspectCritique as AspectCriticMetric
        except ImportError:
            AspectCriticMetric = None

    metrics = [faithfulness, answer_relevancy,
               context_precision, context_recall]
    if AspectCriticMetric is not None:
        metrics.append(AspectCriticMetric(name="cites_sources",
                                          definition=aspect_definition))

    hf_dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in run_outputs],
        "answer":       [r["answer"]       for r in run_outputs],
        "contexts":     [r["contexts"]     for r in run_outputs],
        "ground_truth": [r["ground_truth"] for r in run_outputs],
    })

    print("[ragas] calling ragas.evaluate() — issuing LLM-as-judge calls...")
    scores = ragas_evaluate(hf_dataset, metrics=metrics)
    df = scores.to_pandas()
    aspect_col = next((c for c in df.columns
                       if c in ("cites_sources", "aspect_critic")), None)
    summary = {
        "framework":          "ragas",
        "judge_model":        judge_model,
        "faithfulness":       float(df["faithfulness"].mean()),
        "answer_relevancy":   float(df["answer_relevancy"].mean()),
        "context_precision":  float(df["context_precision"].mean()),
        "context_recall":     float(df["context_recall"].mean()),
        "aspect_critic":      (float(df[aspect_col].mean())
                               if aspect_col else 0.0),
        "aspect_critic_definition": aspect_definition,
        "judge_calls":        len(run_outputs) * len(metrics),
        "rows":               df.to_dict(orient="records"),
    }
    return summary


FAITHFULNESS_PROMPT = (
    "You are an evaluator. The answer below was produced by a system "
    "that should ONLY use the provided context. Score how faithful the "
    "answer is to the context on a 0.0-1.0 scale.\n\n"
    "Context:\n{context}\n\nAnswer:\n{answer}\n\n"
    "Respond with ONLY the number."
)
ANSWER_RELEVANCY_PROMPT = (
    "You are an evaluator. Score how directly the answer addresses "
    "the question on a 0.0-1.0 scale.\n\n"
    "Question: {question}\n\nAnswer: {answer}\n\n"
    "Respond with ONLY the number."
)
CONTEXT_PRECISION_PROMPT = (
    "You are an evaluator. Score how many of the chunks below are "
    "RELEVANT to answering the question on a 0.0-1.0 scale.\n\n"
    "Question: {question}\n\nChunks:\n{contexts}\n\n"
    "Respond with ONLY the number."
)
CONTEXT_RECALL_PROMPT = (
    "You are an evaluator. Score the fraction of the ground-truth "
    "answer that is COVERED by the retrieved chunks on a 0.0-1.0 "
    "scale.\n\nGround truth: {ground_truth}\n\nChunks:\n{contexts}\n\n"
    "Respond with ONLY the number."
)
ASPECT_CRITIC_PROMPT = (
    "You are an evaluator applying a CUSTOM rubric.\n\n"
    "Rubric:\n{rubric}\n\n"
    "Question: {question}\n\nAnswer: {answer}\n\n"
    "Return ONLY '1' if the answer satisfies the rubric or '0' if it does not."
)


def _parse_score(text: str) -> float:
    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)", text.strip())
    if not m:
        return 0.0
    try:
        v = float(m.group(1))
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, v))


_anthropic_client: Any = None


def _client() -> Any:
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError("anthropic package not installed") from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def _judge(prompt: str, judge_model: str, max_tokens: int = 12) -> float:
    try:
        resp = _client().messages.create(
            model=judge_model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(blk.text for blk in resp.content
                       if getattr(blk, "type", None) == "text")
        return _parse_score(text)
    except Exception as e:
        print(f"[ragas-fallback] WARN: judge call failed ({e!s}) — score=0.0")
        return 0.0


def _evaluate_with_fallback(run_outputs: List[Dict[str, Any]], *,
                            judge_model: str, aspect_definition: str,
                            show_one_judge_prompt: bool = True
                            ) -> Dict[str, Any]:
    out_rows: List[Dict[str, Any]] = []
    sums = {"faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0,
            "aspect_critic": 0.0}

    for i, r in enumerate(run_outputs, start=1):
        ctx_block = "\n---\n".join(r["contexts"]) or "(none)"
        faith_prompt = FAITHFULNESS_PROMPT.format(
            context=ctx_block, answer=r["answer"])
        if show_one_judge_prompt and i == 1:
            print("\n" + "-" * 78)
            print(" THE JUDGE PROMPT (faithfulness, row 1) — exactly what")
            print(" Ragas sends to the LLM under the hood:")
            print("-" * 78)
            print(faith_prompt[:600]
                  + ("..." if len(faith_prompt) > 600 else ""))
            print("-" * 78 + "\n")

        f  = _judge(faith_prompt, judge_model)
        ar = _judge(ANSWER_RELEVANCY_PROMPT.format(
            question=r["question"], answer=r["answer"]), judge_model)
        cp = _judge(CONTEXT_PRECISION_PROMPT.format(
            question=r["question"], contexts=ctx_block), judge_model)
        cr = _judge(CONTEXT_RECALL_PROMPT.format(
            ground_truth=r["ground_truth"], contexts=ctx_block), judge_model)
        ac = _judge(ASPECT_CRITIC_PROMPT.format(
            rubric=aspect_definition,
            question=r["question"], answer=r["answer"]), judge_model)

        sums["faithfulness"]      += f
        sums["answer_relevancy"]  += ar
        sums["context_precision"] += cp
        sums["context_recall"]    += cr
        sums["aspect_critic"]     += ac
        out_rows.append({**r,
                         "faithfulness": f, "answer_relevancy": ar,
                         "context_precision": cp, "context_recall": cr,
                         "aspect_critic": ac})
        print(f"[ragas-fallback] row {i}/{len(run_outputs)}  "
              f"F={f:.2f} AR={ar:.2f} CP={cp:.2f} "
              f"CR={cr:.2f} AC={ac:.2f}")

    n = max(len(run_outputs), 1)
    return {
        "framework":          "ragas (offline fallback)",
        "judge_model":        judge_model,
        "faithfulness":       sums["faithfulness"]      / n,
        "answer_relevancy":   sums["answer_relevancy"]  / n,
        "context_precision": sums["context_precision"] / n,
        "context_recall":     sums["context_recall"]    / n,
        "aspect_critic":      sums["aspect_critic"]     / n,
        "aspect_critic_definition": aspect_definition,
        "judge_calls":        n * 5,
        "rows":               out_rows,
    }
