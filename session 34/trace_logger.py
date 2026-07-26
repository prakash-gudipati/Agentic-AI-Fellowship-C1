"""
Session 34 — trace_logger.py

ANSI-coloured trace printers for the multi-agent demos. demo.py wires
on_event=handle_event into the Orchestrator so every interesting moment
prints a labelled line.
"""

from __future__ import annotations

from typing import Any, Dict


_RED = "\033[91m"
_MINT = "\033[92m"
_AMBER = "\033[93m"
_BLUE = "\033[94m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{_RESET}"


def print_section(label: str) -> None:
    bar = "─" * (len(label) + 4)
    print(f"\n{_c(bar, _RED)}")
    print(f"{_c('  ' + label, _BOLD)}")
    print(f"{_c(bar, _RED)}")


def print_subheader(label: str) -> None:
    print(f"\n{_c('▸ ' + label, _AMBER)}")


def print_user_request(q: str) -> None:
    print(f"\n{_c('USER:', _BLUE)}  {q}")


def print_final_answer(a: str) -> None:
    print(f"\n{_c('ANSWER:', _MINT)}")
    print(_indent(a, 2))


def print_scratchpad(label: str, summary_text: str) -> None:
    print(f"\n{_c('SCRATCHPAD ' + label, _DIM)}")
    print(summary_text)


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())


def handle_event(kind: str, payload: Dict[str, Any]) -> None:
    """One labelled line per interesting orchestrator/worker moment."""

    if kind == "round_start":
        rn = payload.get("round_no")
        print(f"\n{_c(f'── Round {rn} ──', _AMBER)}")
        return

    if kind == "orchestrator_decision":
        worker = payload.get("worker")
        instr = payload.get("instruction")
        if payload.get("done"):
            print(f"  {_c('ORCHESTRATOR:', _MINT)} done — publishing FINAL_ANSWER")
        else:
            print(f"  {_c('ORCHESTRATOR:', _AMBER)} → {worker}  ({instr})")
        return

    if kind == "worker_started":
        print(f"  {_c('WORKER  ', _BLUE)} {payload.get('worker')} running…")
        return

    if kind == "worker_finished":
        worker = payload.get("worker")
        summary = payload.get("summary", "")
        print(f"  {_c('WORKER  ', _MINT)} {worker} done — {summary}")
        return


    if kind == "scratchpad_write":
        agent = payload.get("agent")
        section = payload.get("section")
        summary = payload.get("summary", "")
        print(f"    {_c('scratchpad', _DIM)} [{agent} -> {section}] {summary}")
        return

    if kind == "critique_verdict":
        accepted = payload.get("accepted")
        if accepted:
            print(f"  {_c('CRITIQUE: ACCEPT', _MINT)}")
        else:
            issues = payload.get("issues") or []
            print(f"  {_c('CRITIQUE: REVISE', _RED)} ({len(issues)} issues)")
            for issue in issues[:3]:
                print(f"    . {issue}")
        return

    if kind == "termination":
        reason = payload.get("reason")
        rounds = payload.get("rounds")
        colour = _MINT if reason in ("all_done", "quality_met") else _RED
        print(f"\n  {_c('TERMINATED:', colour)} {reason} after {rounds} round(s)")
        return

    print(f"  . {kind}: {payload}")
