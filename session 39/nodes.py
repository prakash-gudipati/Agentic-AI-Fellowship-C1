"""
Session 39 — nodes.py

THE NODES. Same idea as S38 — a node takes the State and returns a partial dict
of what it changed — but four things are new this session, and each one is the
mechanism behind one of the session's headline ideas:

    recall_node        — reads the long-term STORE (cross-thread memory)
    plan_node          — writes a human-reviewable PLAN
    human_review_node  — calls interrupt(): the workflow PAUSES here for a human
    save_node          — writes the finished report back to the STORE

The S38 research loop (analyze -> search -> evaluate -> write_report) is carried
forward in spirit; analyze now also reads the approved plan.

Two LangGraph mechanics to notice:
  * A node can receive the Store by declaring a `store: BaseStore` keyword
    parameter — LangGraph injects it because we compiled the graph with a store.
  * `interrupt(payload)` THROWS a special signal that LangGraph catches. It
    saves the State, surfaces `payload` to the caller, and stops. When you
    resume with Command(resume=value), `interrupt(...)` RETURNS `value` and the
    node runs again from the top. (So keep code before interrupt() idempotent.)
"""

from __future__ import annotations

from langgraph.store.base import BaseStore
from langgraph.types import interrupt

import llm
import search_tools
from persistence import REPORTS_NAMESPACE
from state import ResearchState


def recall_node(state: ResearchState, *, store: BaseStore) -> dict:
    """Read long-term memory BEFORE planning.

    Looks in the cross-thread Store for an earlier report on a similar question.
    If found, it becomes `prior_context`, which the planner is allowed to build
    on. This is the Store doing what the checkpointer cannot: carrying knowledge
    from one run (thread) into a different one.
    """
    # store.search returns every item under the namespace. We pick the most
    # word-overlapping past question ourselves — deterministic, and it does not
    # require configuring a semantic index on the Store (which would need an
    # embedder). A production store can do true semantic recall; the API is the
    # same, only the matching gets smarter.
    items = store.search(REPORTS_NAMESPACE, limit=20)
    best = _best_match(state["question"], items)
    if best is not None:
        prior = best.value.get("report", "")
        return {"prior_context": f"(From an earlier run on a related question)\n{prior}"}
    return {"prior_context": ""}


def _best_match(question: str, items: list):
    """Return the stored item whose question shares the most words with this
    one, or None if nothing overlaps. Plain set-overlap, no model."""
    q_words = {w for w in question.lower().split() if len(w) > 3}
    best, best_overlap = None, 0
    for item in items:
        past_q = str(item.value.get("question", "")).lower()
        overlap = len(q_words & {w for w in past_q.split() if len(w) > 3})
        if overlap > best_overlap:
            best, best_overlap = item, overlap
    return best


def plan_node(state: ResearchState) -> dict:
    """Write the short research plan the human will review."""
    plan = llm.make_plan(state["question"], state.get("prior_context", ""))
    return {"plan": plan}


def human_review_node(state: ResearchState) -> dict:
    """THE HUMAN-IN-THE-LOOP PAUSE.

    interrupt() stops the whole workflow and hands the payload back to whoever
    invoked the graph. The run is now checkpointed and paused — it can sit here
    for a second or a week. When the caller resumes with Command(resume=DECISION),
    interrupt() RETURNS that DECISION and this node finishes.

    DECISION contract (kept deliberately simple for students):
      * "approve"                  -> use the proposed plan as-is
      * any other non-empty string -> treat it as the human's EDITED plan
    """
    decision = interrupt({
        "proposed_plan": state["plan"],
        "instructions": "Reply 'approve' to accept, or send edited plan text to override.",
        "question": state["question"],
    })

    decision = (decision or "").strip()
    if decision.lower() in ("", "approve", "approved", "yes", "ok"):
        approved = state["plan"]
    else:
        approved = decision
    return {"approved_plan": approved}


def analyze_query_node(state: ResearchState) -> dict:
    """Turn the APPROVED plan + question into the first search query."""
    query = llm.analyze_query(state["question"], state.get("approved_plan", ""))
    return {"refined_query": query}


def search_node(state: ResearchState) -> dict:
    """Run one web search. Returns ONLY this attempt's new hits; the
    accumulate reducer (operator.add) appends them to earlier loops' hits."""
    new_results = search_tools.web_search(state["refined_query"], max_results=4)
    attempts = state.get("attempts", 0) + 1
    return {"search_results": new_results, "attempts": attempts}


def evaluate_node(state: ResearchState) -> dict:
    """The QUALITY GATE (S38). Score all results so far, record the verdict,
    and stash a refined query for the next loop in case we fail."""
    verdict = llm.evaluate_results(state["question"], state["search_results"])
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
    """Write the final grounded report from the surviving search results."""
    report = llm.write_report(state["question"], state["search_results"])
    return {"report": report}


def save_node(state: ResearchState, *, store: BaseStore) -> dict:
    """Write the finished report to long-term memory AFTER the report is done.

    Now the next run's recall_node can find it. This is the write half of the
    Store's cross-thread memory — the read half is recall_node above.
    """
    store.put(
        REPORTS_NAMESPACE,
        _memory_key(state["question"]),
        {"question": state["question"], "report": state.get("report", ""),
         "quality": state.get("quality_score", 0.0)},
    )
    return {}


def _memory_key(question: str) -> str:
    """A stable-ish key for a question (lowercased first words)."""
    return "-".join(question.lower().split()[:6]) or "untitled"
