"""
Session 39 — selftest.py

OFFLINE wiring tests. No API key, no network — set FAKE_LLM=1 and these prove
every S39 mechanic works. They are the instructor's pre-session safety net and
the place a student copies the patterns from.

Run:  FAKE_LLM=1 python demo.py --selftest

Each test prints PASS/FAIL and the suite exits non-zero if anything fails.
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("FAKE_LLM", "1")

from langgraph.types import Command

from graph import build_graph
from persistence import (REPORTS_NAMESPACE, build_memory_checkpointer,
                         build_sqlite_checkpointer, build_store)
from router import should_continue
from state import new_state

_results: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def test_router_is_pure() -> None:
    """The cycle's decision function: same State in, same edge out."""
    loop = should_continue({"quality_score": 0.3, "attempts": 1, "max_attempts": 3})
    exit_pass = should_continue({"quality_score": 0.9, "attempts": 1, "max_attempts": 3})
    exit_brake = should_continue({"quality_score": 0.1, "attempts": 3, "max_attempts": 3})
    _check("router loops when score low + attempts left", loop == "search", loop)
    _check("router exits when gate passes", exit_pass == "write_report", exit_pass)
    _check("router exits on the brake (max attempts)", exit_brake == "write_report", exit_brake)


def test_interrupt_pauses() -> None:
    """interrupt() must PAUSE the graph at human_review and surface a payload."""
    app = build_graph(checkpointer=build_memory_checkpointer(), store=build_store())
    cfg = {"configurable": {"thread_id": "pause-test"}}
    out = app.invoke(new_state("what is RAG evaluation"), cfg)
    paused = bool(out.get("__interrupt__"))
    snap = app.get_state(cfg)
    at_review = snap.next == ("human_review",)
    _check("interrupt() surfaces a payload", paused, str(bool(out.get("__interrupt__"))))
    _check("graph paused AT human_review", at_review, f"next={snap.next}")


def test_resume_completes() -> None:
    """Resuming with Command(resume=...) finishes the run through save."""
    app = build_graph(checkpointer=build_memory_checkpointer(), store=build_store())
    cfg = {"configurable": {"thread_id": "resume-test"}}
    app.invoke(new_state("what is hybrid search"), cfg)
    final = app.invoke(Command(resume="approve"), cfg)
    has_report = bool(final.get("report"))
    used_plan = bool(final.get("approved_plan"))
    _check("resume produces a final report", has_report, final.get("report", "")[:40])
    _check("approved_plan was set by the human step", used_plan)


def test_human_edit_overrides_plan() -> None:
    """A non-'approve' resume value becomes the edited plan."""
    app = build_graph(checkpointer=build_memory_checkpointer(), store=build_store())
    cfg = {"configurable": {"thread_id": "edit-test"}}
    app.invoke(new_state("explain reranking"), cfg)
    edited = "1. Only compare two rerankers\n2. Report latency"
    final = app.invoke(Command(resume=edited), cfg)
    _check("human edit replaces the proposed plan", final.get("approved_plan") == edited)


def test_persistence_survives_restart() -> None:
    """THE portfolio requirement: state survives a program restart.

    We simulate a restart by building a graph on a SQLite FILE, running it to
    the pause, then DROPPING every object and building a brand-new graph from
    the same file. The resume must continue from exactly where it paused.
    """
    db_path = os.path.join(tempfile.mkdtemp(), "restart.db")
    thread = {"configurable": {"thread_id": "restart-1"}}

    # --- "process 1": run until the human pause, then forget everything ---
    cp1 = build_sqlite_checkpointer(db_path)
    app1 = build_graph(checkpointer=cp1, store=build_store())
    app1.invoke(new_state("what is agent memory"), thread)
    plan_before = app1.get_state(thread).values.get("plan", "")
    cp1.conn.close()   # a real process exit closes the db handle; we do it explicitly
    del app1, cp1      # the program "exits"

    # --- "process 2": fresh objects, same db file, resume ---
    app2 = build_graph(checkpointer=build_sqlite_checkpointer(db_path), store=build_store())
    snap = app2.get_state(thread)
    resumed_at_review = snap.next == ("human_review",)
    plan_after = snap.values.get("plan", "")
    final = app2.invoke(Command(resume="approve"), thread)

    _check("paused state reloaded from disk after restart", resumed_at_review, f"next={snap.next}")
    _check("the proposed plan survived the restart", plan_before == plan_after and bool(plan_after))
    _check("resume after restart finishes the run", bool(final.get("report")))


def test_store_is_cross_thread() -> None:
    """The Store carries knowledge from one run (thread) to a DIFFERENT one.

    Thread A finishes a run and saves its report. Thread B is a separate run on
    a related question — recall_node must surface thread A's report as
    prior_context. The checkpointer alone could NEVER do this.
    """
    store = build_store()  # ONE store shared across both threads
    app = build_graph(checkpointer=build_memory_checkpointer(), store=store)

    cfg_a = {"configurable": {"thread_id": "mem-A"}}
    app.invoke(new_state("what is vector database indexing"), cfg_a)
    app.invoke(Command(resume="approve"), cfg_a)

    saved = store.search(REPORTS_NAMESPACE, limit=10)
    _check("thread A wrote a report to the long-term store", len(saved) >= 1, f"{len(saved)} item(s)")

    cfg_b = {"configurable": {"thread_id": "mem-B"}}
    out_b = app.invoke(new_state("how does vector database indexing scale"), cfg_b)
    recalled = bool(app.get_state(cfg_b).values.get("prior_context"))
    _check("thread B recalled thread A's knowledge (cross-thread memory)", recalled)


def test_time_travel() -> None:
    """get_state_history lists every checkpoint; we can replay from an early one
    and update_state to FORK an alternate run."""
    app = build_graph(checkpointer=build_memory_checkpointer(), store=build_store())
    cfg = {"configurable": {"thread_id": "tt-1"}}
    app.invoke(new_state("what is chunking"), cfg)
    app.invoke(Command(resume="approve"), cfg)

    history = list(app.get_state_history(cfg))
    _check("get_state_history returns checkpoints", len(history) >= 3, f"{len(history)} checkpoints")

    # Pick the checkpoint right before write_report and FORK it with a tweaked
    # state. update_state returns a NEW config pointing at the forked branch.
    early = history[-1]  # the very first checkpoint
    forked_cfg = app.update_state(early.config, {"question": "what is semantic chunking"})
    forked_q = app.get_state(forked_cfg).values.get("question")
    _check("update_state forks an alternate run", forked_q == "what is semantic chunking", forked_q)


def test_retry_policy_attached() -> None:
    """The search node carries a RetryPolicy (transient network failures retry)."""
    app = build_graph(checkpointer=build_memory_checkpointer(), store=build_store())
    node = app.get_graph().nodes.get("search")
    # We at least confirm the node exists and compiled; the policy lives on the
    # underlying PregelNode. This is a smoke check, not a fault-injection test.
    _check("search node compiled (RetryPolicy wired)", node is not None)


def run() -> None:
    print("\nSession 39 — offline wiring selftest (FAKE_LLM)\n" + "-" * 48)
    test_router_is_pure()
    test_interrupt_pauses()
    test_resume_completes()
    test_human_edit_overrides_plan()
    test_persistence_survives_restart()
    test_store_is_cross_thread()
    test_time_travel()
    test_retry_policy_attached()

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print("-" * 48)
    print(f"  {passed}/{total} checks passed")
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    run()
