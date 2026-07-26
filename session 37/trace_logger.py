"""
Session 37 — trace_logger.py

ANSI-coloured printers used by demo.py and callbacks.py. The walkthrough
script refers to these labels by name. Colours degrade gracefully when a
terminal does not honour them (and NO_COLOR is respected).
"""

from __future__ import annotations

import os

_USE_COLOR = os.environ.get("NO_COLOR", "") == ""

_RED = "\033[91m"
_MINT = "\033[92m"
_AMBER = "\033[93m"
_BLUE = "\033[94m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _c(text: str, colour: str) -> str:
    if not _USE_COLOR:
        return text
    return f"{colour}{text}{_RESET}"


def c_red(t: str) -> str:
    return _c(t, _RED)


def c_mint(t: str) -> str:
    return _c(t, _MINT)


def c_amber(t: str) -> str:
    return _c(t, _AMBER)


def c_blue(t: str) -> str:
    return _c(t, _BLUE)


def c_dim(t: str) -> str:
    return _c(t, _DIM)


# ---------------------------------------------------------------------
# Section + block printers
# ---------------------------------------------------------------------


def print_section(label: str) -> None:
    bar = "─" * (len(label) + 4)
    print(f"\n{_c(bar, _RED)}")
    print(f"{_c('  ' + label, _BOLD)}")
    print(f"{_c(bar, _RED)}")


def print_subheader(label: str) -> None:
    print(f"\n{_c('▸ ' + label, _AMBER)}")


def print_note(text: str) -> None:
    print(_c("  " + text, _DIM))


def print_user_question(q: str) -> None:
    print(f"\n{_c('USER:', _BLUE)}  {q}")


def print_final_answer(a: str) -> None:
    print(f"\n{_c('ANSWER:', _MINT)}")
    print(_indent(a, 2))


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())
