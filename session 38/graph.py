"""
Session 38 — graph.py

WIRING THE GRAPH. This is where State + Cycle come together.

We hand LangGraph four nodes and tell it how they connect. Most edges are
plain ("after A, run B"). One edge is CONDITIONAL: after `evaluate`, run the
router (should_continue) and follow whichever edge name it returns. That
conditional edge is the loop.

The shape we build:

        START
          v
    analyze_query
          v
       search  <----------------+
          v                     |
       evaluate                 | "search"  (quality too low, attempts left)
          v                     |
   should_continue ------------>+
          |
          | "write_report"  (good enough, or out of attempts)
          v
     write_report
          v
         END

Compare this to S37: a LangChain chain is a straight line with no way back.
The moment you need to go BACK to an earlier step based on a result, you have
left chains behind and you need a graph.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from nodes import analyze_query_node, evaluate_node, search_node, write_report_node
from router import should_continue
from state import ResearchState


def build_graph():
    """Assemble and compile the Research Workflow graph.

    Returns a compiled app you can .invoke() or .stream() — exactly like any
    other LangChain Runnable.
    """
    graph = StateGraph(ResearchState)

    # 1. Register the four nodes by name.
    graph.add_node("analyze_query", analyze_query_node)
    graph.add_node("search", search_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("write_report", write_report_node)

    # 2. The forward (unconditional) edges.
    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "search")
    graph.add_edge("search", "evaluate")
    graph.add_edge("write_report", END)

    # 3. THE CONDITIONAL EDGE — the loop. After `evaluate`, call should_continue
    #    and route to whichever node name it returns. The third argument maps
    #    each possible return string to a destination node.
    graph.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "search": "search",            # loop back
            "write_report": "write_report",  # exit
        },
    )

    return graph.compile()


def graph_ascii() -> str:
    """Return an ASCII drawing of the graph (needs the `grandalf` package).

    Used by `python demo.py --graph` so students can SEE the loop before they
    read a single line of node code. Drawing the graph does NOT call the model
    or the network — it inspects the compiled topology only.
    """
    app = build_graph()
    try:
        return app.get_graph().draw_ascii()
    except ImportError:
        return ("Install `grandalf` to draw the graph:  pip install grandalf\n"
                "Nodes: analyze_query -> search -> evaluate -> {search | write_report} -> END")
