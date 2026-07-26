"""
Session 32 — trace_logger.py

Tiny terminal pretty-printer for the walkthrough.

Mirrors the S31 helpers and adds three memory-specific printers:
  - print_working_state    — one line summary of buffer occupancy
  - print_eviction         — eviction trace
  - print_summary          — current rolling summary
  - print_semantic_hits    — top-K vector recall hits
  - print_context_ledger   — token breakdown for the assembled prompt

Disable colour with NO_COLOR=1 or by piping to a file (isatty is False).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from memory_types import EvictionEvent, SemanticHit, SummaryNote, Turn

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"


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


def print_section(title: str) -> None:
    print()
    print(_c(f"--- {title} ---", _MAGENTA))


def print_user(text: str) -> None:
    print(_c("USER     >  ", _BLUE) + text)


def print_assistant(text: str) -> None:
    print(_c("ASSISTANT>  ", _GREEN) + text)


def print_working_state(turns: List[Turn], token_count: int, budget: int) -> None:
    pct = (token_count * 100) // budget if budget else 0
    color = _GREEN if pct < 70 else (_YELLOW if pct < 95 else _RED)
    print(
        _c(
            f"  working_memory: turns={len(turns)} "
            f"tokens={token_count}/{budget}  ({pct}%)",
            color,
        )
    )


def print_eviction(events: List[EvictionEvent]) -> None:
    if not events:
        return
    for e in events:
        print(
            _c(
                f"  EVICT  turn={e.turn_id}  policy={e.policy}  reason={e.reason}",
                _YELLOW,
            )
        )


def print_summary(summary: Optional[SummaryNote]) -> None:
    if summary is None or not summary.text:
        print(_c("  summary: (empty)", _DIM))
        return
    covers = len(summary.covers_turn_ids)
    print(_c(f"  summary covers {covers} evicted turn(s):", _BOLD))
    for line in summary.text.splitlines():
        print(_c(f"    {line}", _DIM))


def print_semantic_hits(query: str, hits: List[SemanticHit]) -> None:
    print(_c(f"  semantic recall for: {query!r}", _BOLD))
    if not hits:
        print(_c("    (no hits above threshold)", _DIM))
        return
    for h in hits:
        snippet = h.turn.content.replace("\n", " ")[:90]
        print(
            _c(
                f"    score={h.score:.2f}  [{h.turn.role}]  {snippet}",
                _DIM,
            )
        )


def print_context_ledger(breakdown: dict, total: int) -> None:
    print(_c("  context ledger:", _BOLD))
    for k in ("system", "summary", "semantic_hits", "working", "user"):
        if k in breakdown:
            print(_c(f"    {k:<14} {breakdown[k]:>4} tok", _DIM))
    print(_c(f"    {'TOTAL':<14} {total:>4} tok", _CYAN))
