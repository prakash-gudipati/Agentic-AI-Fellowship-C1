"""
Session 38 — selftest.py

A WIRING TEST, not the demo. It stubs out the two external boundaries — the
model and the search engine — with deterministic fakes, so we can prove the
GRAPH itself is wired correctly without any API key or internet.

Verifies:
  1. The pure router (should_continue) makes the right call in every case.
  2. The compiled graph LOOPS when the gate fails, then EXITS.
  3. The accumulate REDUCER grows search_results across loops (not overwrite).
  4. The max-attempts brake stops an always-failing run.
  5. The prebuilt ReAct agent (ToolNode + tools_condition) compiles + draws.

Run it with:  python demo.py --selftest   (or:  python selftest.py)
"""

from __future__ import annotations

import os

for _var in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING"):
    os.environ[_var] = "false"

import llm
import search_tools
from graph import build_graph
from router import should_continue
from state import QUALITY_THRESHOLD, new_state


def _check(label: str, condition: bool) -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}")
    if not condition:
        raise AssertionError(label)


def test_router() -> None:
    """The router is pure — test every branch directly."""
    print("\n# 1. Router logic (pure, no graph)")
    _check("passes the gate -> write_report",
           should_continue({"quality_score": 0.9, "attempts": 1, "max_attempts": 3}) == "write_report")
    _check("below threshold, attempts left -> search (loop)",
           should_continue({"quality_score": 0.3, "attempts": 1, "max_attempts": 3}) == "search")
    _check("below threshold, out of attempts -> write_report (brake)",
           should_continue({"quality_score": 0.3, "attempts": 3, "max_attempts": 3}) == "write_report")
    _check("exactly at threshold passes",
           should_continue({"quality_score": QUALITY_THRESHOLD, "attempts": 1, "max_attempts": 3}) == "write_report")


def _install_stubs(scores: list) -> None:
    """Replace model + search with deterministic fakes. Each fake search
    returns ONE new hit, so we can prove accumulation across loops."""
    calls = {"i": 0}

    def fake_analyze(question: str) -> str:
        return f"query about {question[:20]}"

    def fake_search(query: str, max_results: int = 4) -> list:
        return [{"title": f"hit-{calls['i'] + 1}", "url": "http://example.test",
                 "content": "stub snippet", "score": 0.5}]

    def fake_evaluate(question: str, results: list) -> dict:
        score = scores[calls["i"]] if calls["i"] < len(scores) else scores[-1]
        calls["i"] += 1
        return {"score": score, "reason": f"stub verdict #{calls['i']}", "refined_query": "refined stub query"}

    def fake_report(question: str, results: list) -> str:
        return "Stub report.\nSources: http://example.test"

    llm.analyze_query = fake_analyze
    llm.evaluate_results = fake_evaluate
    llm.write_report = fake_report
    search_tools.web_search = fake_search


def test_graph_loops_then_exits() -> None:
    print("\n# 2. Graph loops once, then exits on a passing gate")
    _install_stubs(scores=[0.4, 0.85])
    app = build_graph()
    final = app.invoke(new_state("does the loop work?", max_attempts=3))
    _check("looped exactly twice (one fail, one pass)", final["attempts"] == 2)
    _check("final score is the passing score", final["quality_score"] == 0.85)
    _check("a report was written", bool(final["report"]))


def test_reducer_accumulates() -> None:
    print("\n# 3. Accumulate reducer grows state across loops")
    _install_stubs(scores=[0.4, 0.4, 0.9])
    app = build_graph()
    final = app.invoke(new_state("accumulate check", max_attempts=5))
    n = final["attempts"]
    _check(f"search_results accumulated to {n} (one per search, not overwritten)",
           len(final["search_results"]) == n)
    _check(f"history accumulated to {n} records", len(final["history"]) == n)
    _check("kept the FIRST search's hit, not just the last",
           any(h["title"] == "hit-1" for h in final["search_results"]))


def test_brake_stops_runaway() -> None:
    print("\n# 4. Max-attempts brake stops an always-failing run")
    _install_stubs(scores=[0.1, 0.1, 0.1, 0.1, 0.1])
    app = build_graph()
    final = app.invoke(new_state("never satisfiable", max_attempts=3))
    _check("stopped at the ceiling (3 searches)", final["attempts"] == 3)
    _check("still produced a report from what it had", bool(final["report"]))


def test_prebuilt_agent_wiring() -> None:
    print("\n# 5. Prebuilt ReAct agent (ToolNode + tools_condition) wiring")
    from prebuilt_agent import prebuilt_graph_ascii
    drawing = prebuilt_graph_ascii()
    _check("prebuilt graph drew", "agent" in drawing and "tools" in drawing)


def run() -> None:
    print("Session 38 — graph wiring self-test (no API key, no network)")
    test_router()
    test_graph_loops_then_exits()
    test_reducer_accumulates()
    test_brake_stops_runaway()
    test_prebuilt_agent_wiring()
    print("\nAll wiring tests passed. State accumulates, the graph loops, "
          "the brake stops it, and the prebuilt agent wires up.")


if __name__ == "__main__":
    run()
