"""Session 41 (merged into S40) — run an EXPERIMENT across named agent variants.

PROD PATTERN: A-B Experiment — register two (or more) agent variants under names,
run the SAME dataset through each, then pick the winner by gate + pass-rate. This
generalises compare_runs from a 2-run diff to a named experiment you can log.

Callbacks: S35 Best-of-N judging (score N candidates, name a winner) and S33's
quality gate (score before you trust) — here the "candidates" are whole agent
versions and the "judge" is your eval suite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from compare_runs import ComparisonReport, compare, print_comparison
from eval_runner import RunReport, run_suite, print_run


@dataclass
class ExperimentResult:
    """Every variant's run + the head-to-head winner."""
    runs: Dict[str, RunReport] = field(default_factory=dict)
    comparison: ComparisonReport = None
    winner: str = ""


def run_experiment(dataset, variants: Dict[str, object], verbose: bool = True) -> ExperimentResult:
    """Run each variant through the suite, compare, and name the winner.

    `variants` maps a display name -> an agent with .ask(question). The first two
    keys are compared head-to-head (the common A/B case); the winner is whichever
    passes the gate, breaking ties on pass-rate.
    """
    runs: Dict[str, RunReport] = {}
    for name, agent in variants.items():
        report = run_suite(agent, dataset, variant=name)
        runs[name] = report
        if verbose:
            print(f"  >> ran {name}: {report.passed}/{report.total} pass, "
                  f"gate={'PASS' if report.gate_pass else 'FAIL'}")

    names: List[str] = list(variants.keys())
    cmp = compare(runs[names[0]], runs[names[1]])
    if verbose:
        print()
        print_comparison(cmp)

    return ExperimentResult(runs=runs, comparison=cmp, winner=cmp.winner)
