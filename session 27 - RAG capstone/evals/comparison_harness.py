"""
Session 27 -- evals/comparison_harness.py

THE SIGNATURE ARTIFACT OF THE CAPSTONE.

Runs N retrieval strategies against the SAME golden set, calls
`harness_ragas.evaluate(...)` on each, ranks them by composite score,
prints a ranked table.

THE COMPARISON CONTRACT
-----------------------
"Three retrievers, ten golden rows, one defended pick."

You hand this harness a list of `RetrieverConfig` -- each has a name
and a factory function that returns a fresh `RAGPipeline`. It runs
the Ragas DEEP eval against every config, against the same golden
rows, and tells you which one wins. The composite score is a
weighted average of the four primary Ragas metrics:

    composite = 0.4 * faithfulness
              + 0.3 * answer_relevancy
              + 0.3 * context_recall

(`context_precision` is informative but is dominated by `recall` for
the capstone's ranking purposes; we keep it in the per-row table.)

The capstone's defence README cites the ranked table. The pick you
ship in production is row 1 of this ranking. THAT is the pattern
S27 names today.

USAGE
-----
    from evals.comparison_harness import compare_retrievers, RetrieverConfig
    from pipeline import RAGPipeline
    from retrievers.similarity import SimilarityRetriever
    from retrievers.mmr import MMRRetriever          # if present
    from retrievers.hybrid import HybridRetriever    # if present

    def factory_similarity():
        return RAGPipeline(SimilarityRetriever(index), k=4)
    def factory_mmr():
        return RAGPipeline(MMRRetriever(index), k=4)
    def factory_hybrid():
        return RAGPipeline(HybridRetriever(index), k=4)

    configs = [
        RetrieverConfig(name="similarity", factory=factory_similarity),
        RetrieverConfig(name="mmr",         factory=factory_mmr),
        RetrieverConfig(name="hybrid",      factory=factory_hybrid),
    ]

    result = compare_retrievers(configs, golden_rows)
    print(result.summary_table())
    print(f"defended pick: {result.winner_name}")

PRODUCTION PATTERNS NAMED TODAY (S27):
  - comparison-driven retrieval choice              (NEW)
  - composite score as the single ranking number    (NEW)
  - defended pick (numbers, not opinion)            (NEW)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .golden_dataset import GoldenRow, load_golden_dataset
from . import harness_ragas


# Composite score weights. Override per-call by passing `weights=`.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "faithfulness":     0.40,
    "answer_relevancy": 0.30,
    "context_recall":   0.30,
}


@dataclass
class RetrieverConfig:
    """One retrieval combo entered into the comparison.

    `name` shows up in the ranked table; pick something descriptive
    like "mmr-k=4-lambda=0.6" rather than "config_2".

    `factory` is a zero-arg callable that returns a fresh `RAGPipeline`.
    We re-build the pipeline per config so each one gets its own
    LangSmith trace tree.
    """
    name: str
    factory: Callable[[], Any]
    notes: str = ""


@dataclass
class CombatantScore:
    """One retriever's score line in the comparison."""
    name:               str
    faithfulness:       float
    answer_relevancy:   float
    context_precision:  float
    context_recall:     float
    aspect_critic:      float
    composite:          float
    notes:              str = ""
    framework:          str = "ragas"


@dataclass
class ComparisonResult:
    """The full output of one comparison run."""
    rows:           List[CombatantScore] = field(default_factory=list)
    weights:        Dict[str, float] = field(default_factory=dict)
    golden_n:       int = 0
    winner_name:    str = ""

    def summary_table(self) -> str:
        """A printable ranked table. Used in the capstone walkthrough."""
        if not self.rows:
            return "(no comparison rows)"
        # Sort by composite, descending.
        ordered = sorted(self.rows, key=lambda r: -r.composite)
        header = (f"  {'rank':>4s}  {'retriever':<24s}  "
                  f"{'COMP':>6s}  {'F':>6s}  {'AR':>6s}  "
                  f"{'CP':>6s}  {'CR':>6s}  {'AC':>6s}")
        sep = "  " + "-" * (len(header) - 2)
        lines = [
            "",
            "=" * 84,
            f"  COMPARISON HARNESS  -  ranked by composite score",
            f"  weights: {self.weights}",
            f"  golden rows: {self.golden_n}",
            "=" * 84,
            header,
            sep,
        ]
        for i, r in enumerate(ordered, start=1):
            lines.append(
                f"  {i:>4d}  {r.name:<24s}  "
                f"{r.composite:>6.3f}  {r.faithfulness:>6.3f}  "
                f"{r.answer_relevancy:>6.3f}  "
                f"{r.context_precision:>6.3f}  "
                f"{r.context_recall:>6.3f}  "
                f"{r.aspect_critic:>6.3f}"
            )
        lines.append(sep)
        winner = ordered[0]
        lines.append(f"  WINNER: {winner.name}  "
                     f"(composite={winner.composite:.3f})")
        lines.append("=" * 84)
        return "\n".join(lines)


def _composite(scores: Dict[str, Any],
               weights: Dict[str, float]) -> float:
    """Compute the composite score from a Ragas summary dict."""
    total = 0.0
    for metric, weight in weights.items():
        value = float(scores.get(metric, 0.0) or 0.0)
        total += weight * value
    return total


def compare_retrievers(configs: List[RetrieverConfig],
                       rows: Optional[List[GoldenRow]] = None,
                       *,
                       weights: Optional[Dict[str, float]] = None,
                       judge_model: str = "claude-haiku-4-5-20251001",
                       verbose: bool = True) -> ComparisonResult:
    """Run every config against the same golden set, rank by composite.

    Args:
        configs:      the retrieval combos to compare.
        rows:         the golden rows. Defaults to the on-disk set.
        weights:      composite weights. Defaults to DEFAULT_WEIGHTS.
        judge_model:  LLM-as-judge model for Ragas.

    Returns:
        ComparisonResult -- prints via `.summary_table()`.
    """
    rows = rows or load_golden_dataset()
    weights = weights or dict(DEFAULT_WEIGHTS)

    if verbose:
        print()
        print("=" * 84)
        print("  COMPARISON HARNESS  -  starting")
        print("=" * 84)
        print(f"  combatants:  {[c.name for c in configs]}")
        print(f"  golden rows: {len(rows)}")
        print(f"  weights:     {weights}")
        print(f"  judge model: {judge_model}")

    combatants: List[CombatantScore] = []
    for i, config in enumerate(configs, start=1):
        if verbose:
            print()
            print(f"  [{i}/{len(configs)}] running {config.name}...")
        pipeline = config.factory()
        try:
            summary = harness_ragas.evaluate(
                pipeline, rows,
                judge_model=judge_model,
                show_one_judge_prompt=False,
                verbose=False,
            )
        except Exception as e:
            print(f"  [{i}/{len(configs)}] WARN: {config.name} failed "
                  f"({type(e).__name__}: {e!s}) -- skipping.")
            continue

        composite = _composite(summary, weights)
        combatants.append(CombatantScore(
            name=config.name,
            faithfulness=float(summary.get("faithfulness", 0.0) or 0.0),
            answer_relevancy=float(summary.get("answer_relevancy", 0.0) or 0.0),
            context_precision=float(summary.get("context_precision", 0.0) or 0.0),
            context_recall=float(summary.get("context_recall", 0.0) or 0.0),
            aspect_critic=float(summary.get("aspect_critic", 0.0) or 0.0),
            composite=composite,
            notes=config.notes,
            framework=str(summary.get("framework", "ragas")),
        ))

        if verbose:
            print(f"  [{i}/{len(configs)}] {config.name}  "
                  f"composite={composite:.3f}  "
                  f"F={combatants[-1].faithfulness:.2f} "
                  f"AR={combatants[-1].answer_relevancy:.2f} "
                  f"CR={combatants[-1].context_recall:.2f}")

    if not combatants:
        return ComparisonResult(weights=weights, golden_n=len(rows))

    winner = max(combatants, key=lambda r: r.composite)
    result = ComparisonResult(
        rows=combatants,
        weights=weights,
        golden_n=len(rows),
        winner_name=winner.name,
    )
    if verbose:
        print(result.summary_table())
    return result


if __name__ == "__main__":
    # Tiny smoke test using a fake harness shim (no LLM calls).
    print("[comparison] use compare_retrievers() with RetrieverConfigs.")
    print("[comparison] see demo.py for the capstone walkthrough.")
