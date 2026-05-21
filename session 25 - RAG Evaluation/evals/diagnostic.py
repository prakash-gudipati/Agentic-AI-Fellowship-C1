"""
Session 25 — evals/diagnostic.py

THE DIAGNOSTIC READING.

A low score is not a verdict. It is a clue. The shape of the scores
across metrics tells you WHICH layer of the pipeline to fix.

This module reads a Ragas score dict and produces:
  - the weakest layer (retrieval / generation / rubric)
  - the most-likely root cause
  - a specific, actionable fix to try next

The rules are deliberately simple — production teams write longer
diagnostic trees, but the FOUR rules below catch most failures students
will hit in Phase 4:

  rule 1   context_recall LOW and context_precision OK
           → retriever missing relevant chunks; raise k, switch to
             hybrid (S23), or improve chunking (S21)

  rule 2   context_precision LOW and context_recall OK
           → retriever returning junk; add reranker (S23) or tighten
             chunk size

  rule 3   faithfulness LOW and context_precision OK
           → answer is hallucinating despite having the right context;
             tighten the system prompt, lower temperature, demand
             citations explicitly

  rule 4   answer_relevancy LOW and faithfulness OK
           → answer is grounded but dodging the question; rewrite
             the user prompt to push for a direct answer

Safety scoring (bias, toxicity) and dedicated hallucination scoring
move to S41 with DeepEval. For S25, the Ragas faithfulness metric
covers the "is the answer grounded?" question end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ── Tunable thresholds — the bar the team agrees to ship at ────────────────

THRESHOLDS = {
    "faithfulness":         0.80,
    "answer_relevancy":     0.75,
    "context_precision":    0.70,
    "context_recall":       0.70,
    "aspect_critic":        0.60,
}


@dataclass
class Finding:
    """One diagnostic finding — what's weak, the root cause, the fix."""
    severity: str    # "critical" / "warning" / "ok"
    layer: str       # "retrieval" / "generation" / "rubric"
    metric: str
    score: float
    threshold: float
    root_cause: str
    fix: str


def read_scores(ragas_summary: Dict[str, Any],
                _deepeval_summary: Optional[Dict[str, Any]] = None
                ) -> List[Finding]:
    """Read a Ragas score dict and emit diagnostic findings.

    The second argument is accepted for backwards compatibility and
    ignored — DeepEval is deferred to S41/S42.
    """
    findings: List[Finding] = []

    def _check(metric: str, score: float, layer: str,
               root_cause: str, fix: str) -> None:
        threshold = THRESHOLDS.get(metric, 0.7)
        if score >= threshold:
            findings.append(Finding(
                severity="ok", layer=layer, metric=metric,
                score=score, threshold=threshold,
                root_cause="meets threshold",
                fix="hold the line, monitor for regressions",
            ))
            return
        sev = "critical" if score < threshold * 0.7 else "warning"
        findings.append(Finding(
            severity=sev, layer=layer, metric=metric,
            score=score, threshold=threshold,
            root_cause=root_cause, fix=fix,
        ))

    # ── Retrieval layer ──────────────────────────────────────────────────
    cp = ragas_summary.get("context_precision", 0.0)
    cr = ragas_summary.get("context_recall", 0.0)
    _check("context_precision", cp, layer="retrieval",
           root_cause="retriever is returning chunks that don't help",
           fix="Add a reranker (S23) or tighten the chunk size (S21). "
               "If using hybrid, try lowering the BM25 weight.")
    _check("context_recall", cr, layer="retrieval",
           root_cause="retriever is MISSING relevant chunks",
           fix="Raise top-K, switch to hybrid retrieval (S23), "
               "or try parent-child / contextual chunks (S24).")

    # ── Generation layer ─────────────────────────────────────────────────
    f = ragas_summary.get("faithfulness", 0.0)
    ar = ragas_summary.get("answer_relevancy", 0.0)
    if f < THRESHOLDS["faithfulness"] and cp >= THRESHOLDS["context_precision"]:
        # Overwrite the generic faithfulness check with a sharper one
        # noting that the retrieval looked fine.
        findings.append(Finding(
            severity="critical", layer="generation",
            metric="faithfulness", score=f,
            threshold=THRESHOLDS["faithfulness"],
            root_cause=("context is fine but the model is hallucinating "
                        "anyway — generation layer issue"),
            fix="Tighten the system prompt — explicitly forbid claims "
                "outside the context. Lower temperature. Demand "
                "citations. If still bad, upgrade the answer model.",
        ))
    else:
        _check("faithfulness", f, layer="generation",
               root_cause="answer drifts off the context",
               fix="Strengthen the system prompt's grounding rule; "
                   "verify retrieval is returning relevant chunks first.")
    _check("answer_relevancy", ar, layer="generation",
           root_cause="answer doesn't directly address the question",
           fix="Reword the user prompt; push for a direct answer. "
               "If the model is dodging despite a clear prompt, "
               "consider a stronger model.")

    # ── Domain rubric ────────────────────────────────────────────────────
    ac = ragas_summary.get("aspect_critic", 0.0)
    _check("aspect_critic", ac, layer="rubric",
           root_cause="custom rubric not satisfied",
           fix="Inspect the AspectCritic definition — is the rubric "
               "phrased clearly? If yes, the answer prompt needs to "
               "explicitly encourage what the rubric is checking for.")

    return findings


def print_report(findings: List[Finding]) -> None:
    """Pretty-print the findings as a diagnostic reading."""
    print()
    print("=" * 78)
    print("  DIAGNOSTIC READING")
    print("=" * 78)
    critical = [f for f in findings if f.severity == "critical"]
    warnings = [f for f in findings if f.severity == "warning"]
    ok = [f for f in findings if f.severity == "ok"]

    if critical:
        print("\nCRITICAL (fix before shipping):")
        for f in critical:
            _print_finding(f)
    if warnings:
        print("\nWARNING (investigate this PR):")
        for f in warnings:
            _print_finding(f)
    if not critical and not warnings:
        print("\nAll metrics above threshold. Ship it. "
              "Add today's run to the score log and move on.")

    if ok:
        print(f"\n{len(ok)} metric(s) at or above threshold.")


def _print_finding(f: Finding) -> None:
    print(f"  [{f.layer:>10s}]  {f.metric:>22s} = {f.score:.2f}  "
          f"(threshold {f.threshold:.2f})")
    print(f"               root cause: {f.root_cause}")
    print(f"               fix       : {f.fix}")


if __name__ == "__main__":
    # Smoke test: pretend Ragas reports a recall problem.
    fake_ragas = {
        "faithfulness":      0.88,
        "answer_relevancy":  0.82,
        "context_precision": 0.76,
        "context_recall":    0.51,   # weak
        "aspect_critic":     0.40,   # weak
    }
    findings = read_scores(fake_ragas)
    print_report(findings)
