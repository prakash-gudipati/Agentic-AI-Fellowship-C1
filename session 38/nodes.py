"""
Session 38 — nodes.py

THE NODES. A node is a function: it takes the State (the whiteboard) and
returns a partial dict of the keys it changed. LangGraph merges that dict back
into the shared State — using each key's reducer if it has one.

Four nodes make up the Research Workflow:

    analyze_query  — seed the first search query from the question
    search         — run a REAL web search; its hits ACCUMULATE in the state
    evaluate       — the QUALITY GATE: score the results, propose a better query
    write_report   — the exit: write the grounded final answer

Watch how `search_node` and `evaluate_node` return their accumulate boxes.
They return only the NEW item(s). They do NOT read the old list and concatenate
it themselves — the reducer (operator.add) does that. Compare this to the old
hand-written `list(state["history"]) + [record]`: the reducer makes that
boilerplate disappear.
"""

from __future__ import annotations

import llm
import search_tools
from state import ResearchState


def analyze_query_node(state: ResearchState) -> dict:
    """Turn the user's question into the first focused search query."""
    query = llm.analyze_query(state["question"])
    return {"refined_query": query}


def search_node(state: ResearchState) -> dict:
    """Run one REAL web search with whatever query is on the whiteboard.

    Returns ONLY this attempt's new hits. Because `search_results` is an
    accumulate box (Annotated[list, operator.add]), LangGraph APPENDS these to
    the hits from earlier loops — the running list grows, it is not erased.
    """
    query = state["refined_query"]
    new_results = search_tools.web_search(query, max_results=4)
    attempts = state.get("attempts", 0) + 1
    return {"search_results": new_results, "attempts": attempts}


def evaluate_node(state: ResearchState) -> dict:
    """The QUALITY GATE. Score how well ALL results gathered so far answer the
    question, record the score and the reason, and stash a refined query for
    the next loop in case we fail the gate."""
    verdict = llm.evaluate_results(state["question"], state["search_results"])

    # One record for THIS attempt. We return it as a single-item list; the
    # accumulate reducer appends it to the running history for us.
    record = {
        "attempt": state.get("attempts", 0),
        "query": state["refined_query"],
        "num_results": len(state.get("search_results", [])),
        "score": verdict["score"],
        "reason": verdict["reason"],
    }

    return {
        "quality_score": verdict["score"],
        "quality_reason": verdict["reason"],
        "refined_query": verdict["refined_query"],
        "history": [record],
    }


def write_report_node(state: ResearchState) -> dict:
    """The exit node. Write the final grounded report and stop."""
    report = llm.write_report(state["question"], state["search_results"])
    return {"report": report}
