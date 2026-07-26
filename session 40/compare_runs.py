"""Session 41 (merged into S40) — COMPARE two runs: which change was better?

PROD PATTERN: Experiment Tracking / A-B Comparison — you can't improve what you
can't compare. Given two RunReports over the SAME dataset, this diffs them
case-by-case (IMPROVED / REGRESSED / SAME) and names an overall winner by gate
then pass-rate. This is the same "score-then-pick" move as S33's quality gate and
S35's Best-of-N judging, now applied to whole agent versions instead of answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from eval_runner import RunReport


@dataclass
class CaseDelta:
    """How one case changed from run A to run B."""
    id: str
    a_pass: bool
    b_pass: bool
    verdict: str   # "IMPROVED" | "REGRESSED" | "SAME"


@dataclass
class ComparisonReport:
    """The full A-vs-B diff plus a single named winner."""
    a_variant: str
    b_variant: str
    deltas: List[CaseDelta] = field(default_factory=list)
    winner: str = ""
    reason: str = ""


def compare(run_a: RunReport, run_b: RunReport) -> ComparisonReport:
    """Diff two runs case-by-case and pick a winner (gate first, then pass-rate)."""
    by_id_a = {r["id"]: r for r in run_a.rows}
    by_id_b = {r["id"]: r for r in run_b.rows}

    deltas: List[CaseDelta] = []
    for cid in by_id_a:
        if cid not in by_id_b:
            continue  # only compare cases present in BOTH runs
        a_pass = bool(by_id_a[cid]["pass"])
        b_pass = bool(by_id_b[cid]["pass"])
        if a_pass == b_pass:
            verdict = "SAME"
        elif b_pass and not a_pass:
            verdict = "IMPROVED"   # B fixed a case A failed
        else:
            verdict = "REGRESSED"  # B broke a case A passed
        deltas.append(CaseDelta(cid, a_pass, b_pass, verdict))

    # Winner: a passing gate beats a failing one; ties break on pass-rate.
    if run_b.gate_pass != run_a.gate_pass:
        winner = run_b.variant if run_b.gate_pass else run_a.variant
        reason = "passes the eval gate where the other fails"
    elif run_b.score != run_a.score:
        winner = run_b.variant if run_b.score > run_a.score else run_a.variant
        reason = f"higher pass-rate ({max(run_a.score, run_b.score):.0%})"
    else:
        winner = run_a.variant  # genuine tie — keep the incumbent
        reason = "tie — incumbent kept (no measured improvement)"

    return ComparisonReport(run_a.variant, run_b.variant, deltas, winner, reason)


def print_comparison(cmp: ComparisonReport) -> None:
    """Readable A/B summary: per-case verdicts, then the named winner."""
    import trace_logger
    print(f"  A = {cmp.a_variant}   vs   B = {cmp.b_variant}")
    for d in cmp.deltas:
        if d.verdict == "IMPROVED":
            tag = trace_logger._c("32", "IMPROVED ")
        elif d.verdict == "REGRESSED":
            tag = trace_logger._c("31", "REGRESSED")
        else:
            tag = trace_logger._c("90", "SAME     ")
        print(f"    {d.id:4} {tag}  A={d.a_pass!s:5} B={d.b_pass!s:5}")
    win = trace_logger._c("32;1", cmp.winner)
    print(f"\n  WINNER: {win}  ({cmp.reason})")
