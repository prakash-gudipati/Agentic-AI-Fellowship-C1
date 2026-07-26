"""
Session 39 — state.py

THE STATE, carried forward from Session 38 and extended for the
Human-in-the-Loop Research Agent.

In S38 the workflow ran start to finish in one go. In S39 the SAME workflow
learns three new tricks, and every one of them needs the State to be a little
richer:

  1. PERSISTENCE — the State is written to a checkpoint after every node, so a
     run can survive a program restart and resume from exactly where it paused.

  2. HUMAN-IN-THE-LOOP — the workflow now generates a PLAN, then PAUSES and
     shows that plan to a human. The human can approve it or edit it. So we add
     a `plan` box (what the agent proposed) and an `approved_plan` box (what the
     human signed off on).

  3. LONG-TERM MEMORY (the Store) — at the end of a run we save the finished
     report to a cross-run memory; at the start of the next run we recall what
     we already know. `prior_context` holds anything recalled from that memory.

The OVERWRITE vs ACCUMULATE distinction from S38 still governs every box.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict, total=False):
    """The shared whiteboard for the Human-in-the-Loop Research Agent.

    `total=False` means any node may return a partial dict of just the keys it
    changed — LangGraph merges it back using each key's reducer.
    """

    # --- set once, at the start ---
    question: str            # the user's original question, never mutated
    max_attempts: int        # hard ceiling on search loops (the cycle's brake)

    # --- NEW in S39: the human-in-the-loop boxes ---
    plan: str                # the research plan the agent PROPOSED
    approved_plan: str       # the plan the HUMAN approved (possibly edited)

    # --- NEW in S39: long-term memory recalled from the Store ---
    prior_context: str       # what we already knew from earlier runs

    # --- OVERWRITE boxes: each write replaces the last value (from S38) ---
    refined_query: str       # the actual string we send to the search engine
    attempts: int            # how many times we have searched (the loop count)
    quality_score: float     # 0.0-1.0: how well results answer the question
    quality_reason: str      # one sentence: WHY the gate scored it that way
    report: str              # the final written answer

    # --- ACCUMULATE boxes: a reducer ADDS each write to what was there (S38) -
    search_results: Annotated[list, operator.add]
    history: Annotated[list, operator.add]   # one record appended per attempt


# The quality gate threshold, unchanged from S38. Below this, the graph loops.
QUALITY_THRESHOLD: float = 0.7

# Default ceiling on the cycle (S31 Max-Step Ceiling / S33 Retrieval Budget).
DEFAULT_MAX_ATTEMPTS: int = 3


def new_state(question: str, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> ResearchState:
    """Build the starting whiteboard for a fresh research run.

    Note that we do NOT seed plan/approved_plan/report here — those are filled
    by nodes as the workflow advances. The two accumulate boxes start empty.
    """
    return {
        "question": question,
        "max_attempts": max_attempts,
        "plan": "",
        "approved_plan": "",
        "prior_context": "",
        "refined_query": "",
        "attempts": 0,
        "quality_score": 0.0,
        "quality_reason": "",
        "report": "",
        "search_results": [],
        "history": [],
    }
