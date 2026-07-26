"""
Session 33 — trace_logger.py

Pretty-printers for the agentic loop's trace stream.

demo.py wires `on_event` to handle_event so every interesting moment in
the loop renders to the terminal with a clear label. The walkthrough
script asks the instructor to scroll the trace alongside the slides —
the labels here are the ones the script quotes by name.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rag_types import AgenticAnswer, NaiveAnswer


# ----------------------------------------------------------------------------
# ANSI colours (degrade gracefully on Windows terminals that don't honour them)
# ----------------------------------------------------------------------------


_RED = "\033[91m"
_MINT = "\033[92m"
_AMBER = "\033[93m"
_BLUE = "\033[94m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{_RESET}"


# ----------------------------------------------------------------------------
# Section + block printers
# ----------------------------------------------------------------------------


def print_section(label: str) -> None:
    bar = "─" * (len(label) + 4)
    print(f"\n{_c(bar, _RED)}")
    print(f"{_c('  ' + label, _BOLD)}")
    print(f"{_c(bar, _RED)}")


def print_subheader(label: str) -> None:
    print(f"\n{_c('▸ ' + label, _AMBER)}")


def print_user_question(q: str) -> None:
    print(f"\n{_c('USER:', _BLUE)}  {q}")


def print_final_answer(a: str) -> None:
    print(f"\n{_c('ANSWER:', _MINT)}")
    print(_indent(a, 2))


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())


# ----------------------------------------------------------------------------
# Event handler — plugged into AgenticRAG.on_event
# ----------------------------------------------------------------------------


def handle_event(kind: str, payload: Dict[str, Any]) -> None:
    """Render one trace event."""

    if kind == "user_question":
        return  # already printed by demo.py
    if kind == "decomposed":
        subs = payload.get("sub_questions") or []
        print_subheader(f"decomposed into {len(subs)} sub-question(s)")
        for i, s in enumerate(subs, 1):
            print(f"  {i}. {s}")
        return
    if kind == "sub_question_start":
        print_subheader(
            f"sub-question {payload.get('index')}: {payload.get('text')}"
        )
        return
    if kind == "decision":
        kind_v = payload.get("kind")
        if kind_v == "final":
            print(f"  {_c('AGENT:', _MINT)} (no tool call — final answer)")
        else:
            tn = payload.get("tool_name")
            ta = payload.get("tool_args") or {}
            print(
                f"  {_c('AGENT:', _AMBER)} call {tn}("
                f"{_short_args(ta)})"
            )
        return
    if kind == "no_retrieval_decided":
        print(f"  {_c('decision: no retrieval needed', _MINT)}")
        return
    if kind == "retrieval_attempt":
        i = payload.get("index")
        q = payload.get("query")
        chunks = payload.get("chunks") or []
        verdict = payload.get("verdict") or ""
        budget = payload.get("budget_remaining")
        print(
            f"  {_c('retrieve', _AMBER)} #{i}  query={q!r}  "
            f"budget_remaining={budget}"
        )
        for line in chunks:
            print(f"    · {line}")
        if verdict:
            colour = _MINT if "gate: PASS" in verdict else _RED
            print(f"    {_c(verdict, colour)}")
        return
    if kind == "gate_verdict":
        # Already printed as part of retrieval_attempt — skip.
        return
    if kind == "gate_failed_triggers_requery":
        print(
            f"  {_c('gate failed → agent will re-query', _RED)}  "
            f"(budget_remaining={payload.get('budget_remaining')})"
        )
        return
    if kind == "budget_exhausted":
        print(
            f"  {_c('BUDGET EXHAUSTED', _RED)} on sub-question: "
            f"{payload.get('sub_question')}"
        )
        return
    # Unknown event — render generically so nothing is silent.
    print(f"  · {kind}: {payload}")


def _short_args(args: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k, v in args.items():
        sv = repr(v)
        if len(sv) > 60:
            sv = sv[:57] + "..."
        parts.append(f"{k}={sv}")
    return ", ".join(parts)


# ----------------------------------------------------------------------------
# Summary printers — called at the end of a demo
# ----------------------------------------------------------------------------


def print_agentic_summary(ans: AgenticAnswer) -> None:
    print_subheader("agentic run summary")
    print(f"  total retrievals : {ans.total_attempts()}")
    print(f"  total chunks seen: {ans.total_chunks_seen()}")
    print(f"  retrieval budget : {ans.retrieval_budget}")
    print(f"  budget exhausted : {ans.budget_exhausted}")
    print(f"  no-retrieval     : {ans.decided_no_retrieval}")
    if ans.decomposed_sub_questions:
        print(f"  decomposed into  : {len(ans.decomposed_sub_questions)}")
    print(f"  wall time (s)    : {ans.elapsed_seconds:.2f}")


def print_naive_summary(ans: NaiveAnswer) -> None:
    print_subheader("naive run summary")
    print(f"  retrievals       : 1")
    print(f"  chunks retrieved : {len(ans.chunks)}")
    print(f"  wall time (s)    : {ans.elapsed_seconds:.2f}")
