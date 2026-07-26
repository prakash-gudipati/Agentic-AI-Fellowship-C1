"""
Session 38 — demo.py

The walkthrough entry point. Modes:

    python demo.py --graph
        Draw the hand-wired Research Workflow graph. No API key, no network.

    python demo.py --prebuilt-graph
        Draw the PREBUILT ReAct agent graph (ToolNode + tools_condition).
        No API key, no network.

    python demo.py --selftest
        Run the offline wiring tests (stubs the model + search). No API key.

    python demo.py "your research question here"
        Run the REAL hand-wired workflow: real Anthropic model + real web
        search, with a live trace printed node by node.

    python demo.py --prebuilt "your question here"
        Run the REAL prebuilt ReAct agent on the same tools.

Real runs need ANTHROPIC_API_KEY (and ideally TAVILY_API_KEY; without it,
search falls back to keyless DuckDuckGo).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Disable LangSmith / LangChain tracing for this session. We are NOT using a
# hosted observability platform here (that was Session 26's topic). Without
# this, LangChain tries to POST every run to api.smith.langchain.com and prints
# a noisy 403 if no LangSmith key is configured. Hard-set so it wins over any
# global env var.
for _var in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING"):
    os.environ[_var] = "false"


def _load_dotenv() -> None:
    """Tiny .env loader so students don't need python-dotenv installed."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def run_real(question: str) -> None:
    """Run the hand-wired workflow and print a live, node-by-node trace."""
    import trace as tr
    from graph import build_graph
    from state import QUALITY_THRESHOLD, new_state

    app = build_graph()
    tr.banner(f"RESEARCH WORKFLOW  -  {question}")

    final_state: dict = {}
    for step in app.stream(new_state(question)):
        for node_name, update in step.items():
            final_state.update(update)
            tr.node_enter(node_name)
            if node_name == "search":
                tr.search_line(final_state.get("refined_query", ""),
                               len(update.get("search_results", [])),
                               len(final_state.get("search_results", [])))
            elif node_name == "evaluate":
                tr.gate_line(update.get("quality_score", 0.0), QUALITY_THRESHOLD,
                             update.get("quality_reason", ""))
                if update.get("quality_score", 0.0) < QUALITY_THRESHOLD \
                        and final_state.get("attempts", 0) < final_state.get("max_attempts", 3):
                    tr.loop_line(update.get("refined_query", ""))
            elif node_name == "write_report":
                tr.report_line()

    tr.final_report(final_state.get("report", "(no report produced)"))
    print(f"\nSearches run: {final_state.get('attempts', 0)} | "
          f"hits accumulated: {len(final_state.get('search_results', []))} | "
          f"final quality: {final_state.get('quality_score', 0.0):.2f}")


def run_prebuilt(question: str) -> None:
    """Run the prebuilt ReAct agent (ToolNode + tools_condition)."""
    import trace as tr
    from langchain_core.messages import HumanMessage
    from prebuilt_agent import build_prebuilt_agent

    app = build_prebuilt_agent()
    tr.banner(f"PREBUILT ReAct AGENT  -  {question}")
    final = app.invoke({"messages": [HumanMessage(content=question)]})
    for m in final["messages"]:
        kind = m.__class__.__name__.replace("Message", "").upper()
        text = m.content if isinstance(m.content, str) else str(m.content)
        print(f"\n[{kind}] {text[:600]}")
    print(f"\nTotal messages accumulated (add_messages reducer): {len(final['messages'])}")


def main() -> None:
    _load_dotenv()
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    if args[0] == "--graph":
        from graph import graph_ascii
        print(graph_ascii())
        return
    if args[0] == "--prebuilt-graph":
        from prebuilt_agent import prebuilt_graph_ascii
        print(prebuilt_graph_ascii())
        return
    if args[0] == "--selftest":
        import selftest
        selftest.run()
        return
    if args[0] == "--prebuilt":
        run_prebuilt(" ".join(args[1:]))
        return

    run_real(" ".join(args))


if __name__ == "__main__":
    main()
