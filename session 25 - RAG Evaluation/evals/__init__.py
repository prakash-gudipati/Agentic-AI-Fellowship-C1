"""Session 25 — evals package.

THE PATTERN — your pipeline is a function under test.

Modules:

  golden_dataset.json     the source of truth — 10 (question, answer,
                          contexts) triples drawn from the demo corpus.

  harness_custom.py       the hand-rolled "did it answer right?" eval.
                          Baseline for the side-by-side that motivates
                          moving to Ragas.

  harness_ragas.py        the Ragas wrapper. All five Ragas metrics —
                          faithfulness, answer relevancy, context
                          precision, context recall, AspectCritic.
                          Includes judge-model parameter + cost
                          tracking + a visible-prompt teaching mode.

  synthetic_dataset.py    Ragas TestsetGenerator wrapper with an offline
                          fallback.

  diagnostic.py           reads a Ragas score dict and emits the
                          weakest layer, root cause, and fix.

  score_log.py            appends one CSV row per eval run, stamped with
                          a UTC timestamp and the current git commit
                          hash.

Forward reference: DeepEval (HallucinationMetric, BiasMetric, ToxicityMetric,
BaseMetric subclassing, pytest integration via assert_test) is introduced
in S41 (Guardrails + Evals) and S42 (Eval-Driven Development) where its
safety-layer and workflow-integration features land naturally.
"""

from .golden_dataset import load_golden_dataset
from .score_log import append_run, read_runs
from .diagnostic import read_scores, print_report

__all__ = [
    "load_golden_dataset",
    "append_run",
    "read_runs",
    "read_scores",
    "print_report",
]
