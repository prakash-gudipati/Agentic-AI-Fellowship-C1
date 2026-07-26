"""
Session 30 — trace_logger.py

Coloured terminal trace, carried over from Session 29.

Think of this file as the LIVE view — the pretty thing you watch during the
demo. The PERSISTENT view is action_log.py, which writes a JSONL file an
SRE can grep later. Both views matter; they serve different audiences.

Production note: in real systems the live view often points at the same
underlying data the persistent view writes. We are keeping them as
separate concerns here so the teaching is sharp — one file for humans
watching the demo, one file for an SRE on call at 2am.
"""

from __future__ import annotations

import sys


# ── ANSI colour codes (a future DISABLE_COLOR switch is one diff away) ──────
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
    sys.stdout.write(f"{prefix_color}{BOLD}{prefix_label:<16}{RESET}{body}\n")
    sys.stdout.flush()


def log_turn_header(turn_number: int) -> None:
    sys.stdout.write(f"\n{DIM}────── turn {turn_number} ──────{RESET}\n")


def log_thought(text: str) -> None:
    _emit(CYAN, "Thought:", text)


def log_action(tool_name: str, tool_args: dict) -> None:
    _emit(YELLOW, "Action:", f"{tool_name}({tool_args})")


def log_observation(text: str) -> None:
    # Keep observations on one line for the live view — truncate long output.
    one_line = " ".join(text.split())
    if len(one_line) > 220:
        one_line = one_line[:217] + "..."
    _emit(GREEN, "Observation:", one_line)


def log_retry(attempt_number: int, tool_name: str) -> None:
    if attempt_number == 1:
        return
    _emit(MAGENTA, "Retry:", f"{tool_name}: attempt {attempt_number}")


def log_final_answer(text: str) -> None:
    sys.stdout.write(f"\n{BOLD}{GREEN}Final Answer:{RESET} {text}\n")


def log_error(text: str) -> None:
    _emit(RED, "Error:", text)


def log_info(text: str) -> None:
    _emit(BLUE, "Info:", text)
