"""Session 41 (merged into S40) — the EVAL RUNNER: one run = one durable RunReport.

EDD treats every eval run like a test run you can SAVE and COMPARE later. This
module wraps the existing eval_harness.run_eval() (we do NOT re-define the
DeepEval metrics — there is exactly one place they live) and packages the result
into a RunReport dataclass that knows its own timestamp, per-case rows, aggregate
pass count, and the overall gate verdict. Reports save/load to JSON so two runs
(e.g. prompt v1 vs v2) can be diffed by compare_runs.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from eval_harness import run_eval  # the SINGLE source of metric definitions


@dataclass
class RunReport:
    """The saved result of running one dataset against one agent variant."""
    variant: str
    timestamp: str
    rows: List[dict] = field(default_factory=list)   # per-case: id/must_block/blocked/scores/pass
    total: int = 0
    passed: int = 0
    gate_pass: bool = False

    @property
    def score(self) -> float:
        """Aggregate pass-rate (0..1) — the headline number A/B comparison sorts on."""
        return (self.passed / self.total) if self.total else 0.0


def run_suite(agent, dataset, variant: str = "variant") -> RunReport:
    """Run a dataset through an agent and package a RunReport.

    Reuses eval_harness.run_eval (real DeepEval metrics + coverage assertion); this
    function only adds the bookkeeping EDD needs: a name, a timestamp, a pass count.
    """
    rows, gate_pass = run_eval(agent, dataset)
    passed = sum(1 for r in rows if r["pass"])
    return RunReport(
        variant=variant,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rows=rows,
        total=len(rows),
        passed=passed,
        gate_pass=gate_pass,
    )


def save_run(report: RunReport, path) -> None:
    """Persist a RunReport to JSON so it can be reloaded and compared later."""
    Path(path).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def load_run(path) -> RunReport:
    """Reload a RunReport written by save_run()."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunReport(**data)


def print_run(report: RunReport) -> None:
    """One-line-per-case scorecard + the gate verdict for this single run."""
    import trace_logger
    print(f"  variant: {report.variant}   {report.passed}/{report.total} cases pass")
    trace_logger.log_scorecard(report.rows, report.gate_pass)
