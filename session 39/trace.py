"""
Session 39 — trace.py

Console printing for the live walkthrough. Extends S38's labels (NODE, SEARCH,
GATE, LOOP, REPORT) with the S39 ones: RECALL, PLAN, PAUSE, RESUME, SAVE,
TIME-TRAVEL. Colours degrade gracefully and honour NO_COLOR.
"""

from __future__ import annotations

import os

_USE_COLOR = os.environ.get("NO_COLOR", "") == ""
_RED, _MINT, _AMBER, _BLUE, _PURPLE, _DIM, _BOLD, _RESET = (
    "\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[2m", "\033[1m", "\033[0m")


def _c(text: str, colour: str) -> str:
    return text if not _USE_COLOR else f"{colour}{text}{_RESET}"


def banner(text: str) -> None:
    bar = "-" * (len(text) + 4)
    print(f"\n{_c(bar, _RED)}\n{_c('  ' + text, _BOLD)}\n{_c(bar, _RED)}")


def node_enter(name: str) -> None:
    print(f"\n{_c('> NODE', _AMBER)}  {_c(name, _BOLD)}")


def recall_line(found: bool) -> None:
    msg = "found prior knowledge in long-term store" if found else "no prior knowledge — fresh start"
    print(f"  {_c('RECALL', _PURPLE)} {msg}")


def plan_line(plan: str) -> None:
    print(f"  {_c('PLAN', _BLUE)}")
    for line in plan.splitlines():
        print(f"         {line}")


def pause_line(payload: dict) -> None:
    print(f"\n{_c('|| PAUSE (human-in-the-loop)', _AMBER)}")
    print(f"   The workflow is now checkpointed and waiting. Proposed plan:")
    for line in str(payload.get("proposed_plan", "")).splitlines():
        print(f"     {line}")
    print(f"   {_c(payload.get('instructions', ''), _DIM)}")


def resume_line(decision: str) -> None:
    print(f"\n{_c('>> RESUME', _MINT)} human decision: {decision!r}")


def search_line(query: str, new_n: int, total_n: int) -> None:
    print(f"  {_c('SEARCH', _BLUE)}  query={query!r}  ->  +{new_n} new  "
          f"({total_n} total in state)")


def gate_line(score: float, threshold: float, reason: str) -> None:
    verdict = "PASS" if score >= threshold else "FAIL"
    colour = _MINT if verdict == "PASS" else _RED
    print(f"  {_c('GATE', colour)}   score={score:.2f}  (threshold {threshold:.2f})  -> {_c(verdict, colour)}")
    print(f"         {_c(reason, _DIM)}")


def loop_line(next_query: str) -> None:
    print(f"  {_c('LOOP', _AMBER)}   quality too low — searching again with {next_query!r}")


def report_line() -> None:
    print(f"  {_c('REPORT', _MINT)} writing the final grounded answer")


def save_line() -> None:
    print(f"  {_c('SAVE', _PURPLE)}   report written to long-term store (next run can recall it)")


def final_report(text: str) -> None:
    print(f"\n{_c('FINAL REPORT', _MINT)}")
    print("\n".join("  " + line for line in text.splitlines()))


def timetravel_line(text: str) -> None:
    print(f"  {_c('TIME-TRAVEL', _PURPLE)} {text}")
