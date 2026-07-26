"""
Session 39 — graph.py

WIRING THE GRAPH and plugging in PERSISTENCE.

The S38 research loop is still here. S39 adds three nodes around it and, more
importantly, COMPILES the graph with a checkpointer and a store:

        START
          v
       recall            <- reads long-term STORE (cross-thread memory)
          v
        plan             <- proposes a research plan
          v
    human_review         <- interrupt(): PAUSE for human approval  [HITL]
          v
    analyze_query
          v
       search  <--------------------+
          v                         |
       evaluate                     | "search"  (quality low, attempts left)
          v                         |
   should_continue ---------------->+
          | "write_report"
          v
     write_report
          v
        save             <- writes report to long-term STORE
          v
         END

Two compile-time arguments do all the heavy lifting:
    checkpointer=...  -> every node's State snapshot is saved (resumable/HITL)
    store=...         -> recall_node / save_node get a cross-thread memory

And one node gets a RetryPolicy: `search` calls the network, so a transient
failure should be retried automatically rather than crashing the run.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from nodes import (analyze_query_node, evaluate_node, human_review_node,
                   plan_node, recall_node, save_node, search_node,
                   write_report_node)
from router import should_continue
from state import ResearchState


def build_graph(checkpointer=None, store=None):
    """Assemble and compile the Human-in-the-Loop Research Agent.

    Pass a checkpointer to make runs pausable/resumable (REQUIRED for the HITL
    interrupt to work across invocations) and a store for cross-thread memory.
    With neither, you get a plain run that cannot pause — useful only for
    drawing the graph.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("recall", recall_node)
    graph.add_node("plan", plan_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("analyze_query", analyze_query_node)
    # RetryPolicy: retry the network call up to 3 times on a transient error.
    graph.add_node("search", search_node, retry_policy=RetryPolicy(max_attempts=3))
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("write_report", write_report_node)
    graph.add_node("save", save_node)

    graph.add_edge(START, "recall")
    graph.add_edge("recall", "plan")
    graph.add_edge("plan", "human_review")
    graph.add_edge("human_review", "analyze_query")
    graph.add_edge("analyze_query", "search")
    graph.add_edge("search", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        should_continue,
        {"search": "search", "write_report": "write_report"},
    )
    graph.add_edge("write_report", "save")
    graph.add_edge("save", END)

    return graph.compile(checkpointer=checkpointer, store=store)


def graph_ascii() -> str:
    """ASCII drawing of the graph topology. No model, no network, no key."""
    app = build_graph()
    try:
        return app.get_graph().draw_ascii()
    except ImportError:
        return ("Install `grandalf` to draw the graph:  pip install grandalf\n"
                "recall -> plan -> human_review -> analyze_query -> search -> "
                "evaluate -> {search | write_report} -> save -> END")
