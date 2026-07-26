"""
Session 36 — trace_logger.py

ANSI-coloured trace printers used by demo.py. The walkthrough script refers to
these labels by name, so the instructor can point at the screen and say "see
the DISCOVER line, then the TOOL CALL line".

Set NO_COLOR=1 to disable colour (useful when piping a trace to a log file).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List


_USE_COLOR = os.environ.get("NO_COLOR", "") != "1"

_RED = "\033[91m"
_MINT = "\033[92m"
_AMBER = "\033[93m"
_BLUE = "\033[94m"
_GREY = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _c(text: str, colour: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{colour}{text}{_RESET}"


def print_section(label: str) -> None:
    bar = "─" * (len(label) + 4)
    print(f"\n{_c(bar, _RED)}")
    print(f"{_c('  ' + label, _BOLD)}")
    print(f"{_c(bar, _RED)}")


def print_user_question(q: str) -> None:
    print(f"\n{_c('USER:', _BLUE)}  {q}")


def print_final_answer(a: str) -> None:
    print(f"\n{_c('ANSWER:', _MINT)}")
    for line in a.splitlines() or [a]:
        print("  " + line)


def print_tool_catalog(specs: List[Any]) -> None:
    """Print what the server advertised during discovery."""

    print(f"{_c('DISCOVER:', _AMBER)} server advertised {len(specs)} tool(s)")
    for s in specs:
        props = list((s.input_schema or {}).get("properties", {}).keys())
        print(f"  • {_c(s.name, _BOLD)}({', '.join(props)}) — {s.description.splitlines()[0]}")


def handle_event(kind: str, payload: Dict[str, Any]) -> None:
    """Render one trace event emitted by the agent loop."""

    if kind == "discover":
        print(f"  {_c('DISCOVER', _AMBER)}: {payload['count']} tools from MCP server")
    elif kind == "think":
        text = payload.get("text", "").strip()
        if text:
            print(f"  {_c('THINK', _GREY)}: {text}")
    elif kind == "tool_call":
        print(
            f"  {_c('TOOL CALL', _AMBER)} → {_c(payload['name'], _BOLD)}"
            f"  args={payload['args']}  {_c('[over MCP]', _GREY)}"
        )
    elif kind == "observation":
        obs = payload.get("text", "")
        snippet = obs if len(obs) <= 200 else obs[:200] + " …"
        print(f"  {_c('OBSERVE', _MINT)}: {snippet}")
    elif kind == "final":
        pass  # printed by print_final_answer
    elif kind == "stats":
        print(
            f"\n{_c('RUN STATS', _BLUE)}: "
            f"{payload['tool_calls']} tool call(s) over MCP · "
            f"{payload['llm_calls']} LLM call(s)"
        )
    else:
        print(f"  · {kind}: {payload}")
