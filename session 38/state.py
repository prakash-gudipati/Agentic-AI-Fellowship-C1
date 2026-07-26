"""
Session 38 — state.py

THE STATE. The first of the two headline ideas in this session.

In pure Python (S29-S36) you passed variables around by hand. In LangGraph
there is ONE shared object that every node reads from and writes to. That
object is the State — the shared whiteboard in a meeting room.

But there is a second, deeper idea that makes LangGraph click: HOW a write
lands on the whiteboard. There are two kinds of box:

  OVERWRITE box  — a write replaces what was there. (the default)
                   Good for a single current value: the latest score, the
                   current query, the attempt count.

  ACCUMULATE box — a write is ADDED to what was there. You declare this with
                   a "reducer": Annotated[list, operator.add]. LangGraph runs
                   that reducer to merge the old value and the new one.
                   Good for things that GROW: every search hit we have ever
                   found, the running history of attempts.

This is the part beginners miss. Without a reducer, the second search would
ERASE the results of the first. With operator.add as the reducer, the second
search's hits are appended to the first's — the box accumulates across the
loop. That is what makes a stateful, looping workflow actually remember.

(The most famous reducer is `add_messages`, used by LangGraph's MessagesState
to accumulate a chat history. We use that one in prebuilt_agent.py.)
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict, total=False):
    """The shared whiteboard for the Research Workflow.

    Note which boxes OVERWRITE and which ACCUMULATE — that choice is the whole
    lesson. `total=False` means a node may return a partial dict.
    """

    # --- set once, at the start ---
    question: str            # the user's original question, never mutated
    max_attempts: int        # hard ceiling on search loops (the cycle's brake)

    # --- OVERWRITE boxes: each write replaces the last value ---
    refined_query: str       # the actual string we send to the search engine
    attempts: int            # how many times we have searched (the loop count)
    quality_score: float     # 0.0-1.0: how well results answer the question
    quality_reason: str      # one sentence: WHY the gate scored it that way
    report: str              # the final written answer

    # --- ACCUMULATE boxes: a reducer ADDS each write to what was there ---
    # operator.add on two lists concatenates them. So every loop's search hits
    # are APPENDED to the running list instead of erasing it.
    search_results: Annotated[list, operator.add]
    history: Annotated[list, operator.add]   # one record appended per attempt


# The quality gate threshold. If the evaluate node scores the search results
# below this, the graph loops back and searches again with a refined query.
# This is a REAL quality check (an LLM judging sufficiency), not a counter.
QUALITY_THRESHOLD: float = 0.7

# Default ceiling on the cycle. Borrowed straight from S31's Max-Step Ceiling
# and S33's Retrieval Budget: a loop without a brake is a bug, not a feature.
DEFAULT_MAX_ATTEMPTS: int = 3


def new_state(question: str, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> ResearchState:
    """Build the starting whiteboard for a fresh research run.

    The two accumulate boxes start empty; the reducer grows them as the
    workflow loops.
    """
    return {
        "question": question,
        "max_attempts": max_attempts,
        "refined_query": "",
        "attempts": 0,
        "quality_score": 0.0,
        "quality_reason": "",
        "report": "",
        "search_results": [],
        "history": [],
    }
