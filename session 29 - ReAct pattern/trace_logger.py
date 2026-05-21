"""
Session 29 — trace_logger.py

Pretty-print the agent's Thought → Action → Observation trace.

Why does this exist as its own module?
  - In Phase 5 you will be staring at agent traces a LOT. Either the agent
    is brilliant and you want to admire the trace, or it loops forever and
    you need to see exactly where it went wrong. Both cases need readable
    output.
  - Print statements scattered through the loop file become noise. A tiny
    logger that knows the SHAPE of an agent trace produces consistent,
    grep-able output across every agent we build for the rest of the
    course.

This is structured terminal logging, not full observability. S26 (Observability)
plus S31 (Agent action logging) will take this much further. For S29, all we
need is: every loop turn announces itself, every Thought/Action/Observation
is on its own line, with a consistent prefix.
"""

from __future__ import annotations

import sys


# ── ANSI color codes ────────────────────────────────────────────────────────
# Defined as constants so a future production switch (DISABLE_COLOR=1) is one
# diff. The intent matters: colour is a debugging aid, not the data itself.
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"


def _emit(prefix_color: str, prefix_label: str, body: str) -> None:
    """Internal — write one labelled line to stdout."""
    sys.stdout.write(f"{prefix_color}{BOLD}{prefix_label:<14}{RESET}{body}\n")
    sys.stdout.flush()


def log_turn_header(turn_number: int) -> None:
    sys.stdout.write(
        f"\n{DIM}────── turn {turn_number} ──────{RESET}\n"
    )


def log_thought(text: str) -> None:
    _emit(CYAN, "Thought:", text)


def log_action(tool_name: str, tool_input: str) -> None:
    _emit(YELLOW, "Action:", f"{tool_name}[{tool_input}]")


def log_observation(text: str) -> None:
    _emit(GREEN, "Observation:", text)


def log_tool_retry(attempt_number: int, tool_input: str) -> None:
    if attempt_number == 1:
        return  # first attempt is the default — only log retries
    _emit(MAGENTA, "Retry:", f"attempt {attempt_number} with input {tool_input!r}")


def log_final_answer(text: str) -> None:
    sys.stdout.write(f"\n{BOLD}{GREEN}Final Answer:{RESET} {text}\n")


def log_error(text: str) -> None:
    _emit(RED, "Error:", text)


def log_info(text: str) -> None:
    _emit(BLUE, "Info:", text)
