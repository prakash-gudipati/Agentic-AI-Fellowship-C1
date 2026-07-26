"""
Session 39 — router.py

THE CYCLE, carried forward unchanged from S38. After the evaluate node, this
PURE function decides the next node:

    score >= threshold  -> "write_report"   (good enough; exit the loop)
    out of attempts     -> "write_report"   (the brake; never loop forever)
    otherwise           -> "search"         (loop back, try the refined query)

Pure function: same State in, same decision out, no model call. That is why it
is the one piece we unit-test directly in selftest.py.
"""

from __future__ import annotations

from state import QUALITY_THRESHOLD, ResearchState


def should_continue(state: ResearchState) -> str:
    """Decide the next node after the quality gate."""
    score = state.get("quality_score", 0.0)
    attempts = state.get("attempts", 0)
    max_attempts = state.get("max_attempts", 3)

    if score >= QUALITY_THRESHOLD:
        return "write_report"          # the gate passed — good enough
    if attempts >= max_attempts:
        return "write_report"          # the brake — stop looping
    return "search"                    # loop back and search the refined query
