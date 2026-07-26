"""
Session 38 — router.py

THE CYCLE. The second of the two headline ideas in this session.

A chain (S37) only ever flows forward: prompt -> model -> parser -> done. A
GRAPH can loop. This function is what makes the loop possible: after the
evaluate node runs, LangGraph calls should_continue(state) and uses the
returned string to decide which node runs next.

    score >= threshold  -> "write_report"   (good enough; exit the loop)
    out of attempts     -> "write_report"   (the brake; never loop forever)
    otherwise           -> "search"         (loop back, try the refined query)

This is a PURE function: same State in, same decision out. No model call, no
network. That is exactly why it is the one piece of the graph we can unit-test
offline — see selftest.py.
"""

from __future__ import annotations

from state import QUALITY_THRESHOLD, ResearchState


def should_continue(state: ResearchState) -> str:
    """Decide the next node after the quality gate.

    Returns the string name of the edge to follow: "search" to loop, or
    "write_report" to exit. The two exit conditions are deliberately separate
    so the trace can tell you WHY the loop ended.
    """
    score = state.get("quality_score", 0.0)
    attempts = state.get("attempts", 0)
    max_attempts = state.get("max_attempts", 3)

    if score >= QUALITY_THRESHOLD:
        return "write_report"          # the gate passed — good enough
    if attempts >= max_attempts:
        return "write_report"          # the brake — stop looping, report what we have
    return "search"                    # loop back and search the refined query
