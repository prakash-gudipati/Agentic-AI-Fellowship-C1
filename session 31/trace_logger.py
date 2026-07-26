"""
Session 31 — trace_logger.py

Tiny pretty-printer for the terminal walkthrough.

ANSI colors so the instructor's screen-share is readable from the back row.
Everything here is cosmetic — disable it (`USE_COLOR=False`) and the agent
still runs the same. The script only uses two entry points:

  print_plan(plan)       — pretty-prints the Plan the planner produced
  print_step_event(step, result) — one line per executed step
  print_decision(decision) — one line per monitor verdict
  print_block(title, body) — a labelled separator block
"""

from __future__ import annotations

import os
import sys

from plan_types import Plan, Step, StepResult

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}" if USE_COLOR else text


def print_block(title: str, body: str = "") -> None:
    bar = "=" * 78
    print()
    print(_c(bar, _CYAN))
    print(_c(f"  {title}", _BOLD))
    if body:
        print(_c(bar, _CYAN))
        print(body)
    print(_c(bar, _CYAN))


def print_plan(plan: Plan) -> None:
    print(_c(f"GOAL: {plan.goal}", _BOLD))
    if plan.synthesis_hint:
        print(_c(f"SYNTHESIS HINT: {plan.synthesis_hint}", _DIM))
    print(_c("STEPS:", _BOLD))
    for step in plan.steps:
        deps = (
            _c(f"  (after {','.join(step.depends_on)})", _DIM)
            if step.depends_on else ""
        )
        print(f"  {_c(step.id, _BLUE)}{deps}  {step.description}")
        print(_c(f"       tool={step.tool}  args={step.args}", _DIM))


def print_step_event(step: Step, result: StepResult) -> None:
    if result.status == "ok":
        marker = _c("✓", _GREEN)
    elif result.status == "error":
        marker = _c("✗", _RED)
    else:
        marker = _c("·", _YELLOW)
    print(f"  {marker} {_c(step.id, _BLUE)} {step.description}")
    print(_c(f"     observation: {result.observation[:160]}", _DIM))


def print_decision(decision) -> None:
    """Print a MonitorDecision in one labelled line."""

    color = _YELLOW if decision.should_replan else _DIM
    label = "REPLAN" if decision.should_replan else "continue"
    print(_c(f"  monitor: {label} ({decision.source}) — {decision.reason}", color))
