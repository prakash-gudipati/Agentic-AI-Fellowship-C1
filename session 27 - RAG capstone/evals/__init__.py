"""Session 27 -- evals package (carry-over from S25/S26, Ragas-only).

THE PATTERN -- your pipeline is a function under test.

Modules:

  golden_dataset.json       the source of truth -- (question, answer,
                            contexts) triples drawn from the corpus.

  harness_custom.py         the hand-rolled "did it answer right?" eval.

  harness_ragas.py          the Ragas wrapper. All five Ragas metrics --
                            faithfulness, answer relevancy, context
                            precision, context recall, AspectCritic.
                            Throttled to 4 workers + 6 retries to respect
                            Anthropic's 50 RPM tier limit.

  synthetic_dataset.py      Ragas TestsetGenerator wrapper with an offline
                            Anthropic-only fallback.

  diagnostic.py             reads a Ragas score dict and emits the
                            weakest layer, root cause, and fix.

  score_log.py              appends one CSV row per eval run, stamped
                            with a UTC timestamp and the current git
                            commit hash.

  comparison_harness.py     NEW in S27. Runs N retrieval combos against
                            the SAME golden set, ranks them by composite
                            Ragas score, prints a ranked table. The
                            signature artifact of the capstone.

DeepEval is intentionally deferred to S41 (Guardrails) and S42 (EDD).
Phase 4 stays Ragas-only.
"""

from .golden_dataset import load_golden_dataset
from .score_log import append_run, read_runs
from .diagnostic import read_scores, print_report
from .comparison_harness import (compare_retrievers,
                                 RetrieverConfig,
                                 ComparisonResult)

__all__ = [
    "load_golden_dataset",
    "append_run",
    "read_runs",
    "read_scores",
    "print_report",
    "compare_retrievers",
    "RetrieverConfig",
    "ComparisonResult",
]
