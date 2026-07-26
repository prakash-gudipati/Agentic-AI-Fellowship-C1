"""Session 40 — tiny ANSI trace helpers so the screen-share reads as a story.

Honors NO_COLOR (set NO_COLOR=1 to disable). Nothing here affects behaviour —
it only makes guardrail decisions, chain reports, and the eval scorecard legible.
"""
from __future__ import annotations

import os
from typing import List

from guardrail_types import Decision, GuardrailReport, GuardrailResult

_NO_COLOR = bool(os.environ.get("NO_COLOR"))


def _c(code: str, text: str) -> str:
    """Wrap text in an ANSI colour unless NO_COLOR is set."""
    if _NO_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _decision_color(d: Decision) -> str:
    return {"ALLOW": "32", "REDACT": "33", "BLOCK": "31"}[d.value]


def banner(title: str) -> None:
    """Print a demo section banner."""
    line = "=" * 64
    print(_c("36", line))
    print(_c("36;1", f"  {title}"))
    print(_c("36", line))


def log_result(r: GuardrailResult, indent: str = "  ") -> None:
    """Print a single guardrail verdict."""
    tag = _c(_decision_color(r.decision), f"[{r.decision.value:6}]")
    print(f"{indent}{tag} {r.name:24} ({r.severity.value:8}) {r.reason}")


def log_report(report: GuardrailReport) -> None:
    """Print every result in a chain report, then the final decision."""
    print(_c("1", f"  {report.stage.upper()} CHAIN  ({len(report.results)} checks):"))
    for r in report.results:
        log_result(r)
    final = _c(_decision_color(report.final_decision), report.final_decision.value)
    print(f"  -> final: {final}")
    if report.final_decision != Decision.BLOCK:
        print(f"     text: {report.final_text!r}")


def log_turn(question: str, result: dict) -> None:
    """Print a full SafeAgent turn: question, model_called, answer."""
    print(_c("1", "  Q: ") + question)
    print(f"  model_called: {result['model_called']}")
    if result.get("input_report"):
        log_report(result["input_report"])
    if result.get("output_report"):
        log_report(result["output_report"])
    print(_c("32;1", "  A: ") + result["answer"])


def log_scorecard(rows: List[dict], gate_pass: bool) -> None:
    """Print the eval scorecard table + the overall gate verdict."""
    print()
    print(_c("1", "  SCORECARD"))
    header = f"  {'id':4} {'must_block':10} {'blocked':8} {'ground':7} {'safe':6} {'relev':6} {'PASS':5}"
    print(_c("36", header))
    for row in rows:
        ok = _c("32", "PASS") if row["pass"] else _c("31", "FAIL")
        print(f"  {row['id']:4} {str(row['must_block']):10} {str(row['blocked']):8} "
              f"{row['ground']:<7} {row['safe']:<6} {row['relev']:<6} {ok}")
    print()
    verdict = _c("32;1", "GATE: PASS") if gate_pass else _c("31;1", "GATE: FAIL")
    print(f"  {verdict}")
