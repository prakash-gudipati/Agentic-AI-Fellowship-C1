"""
Session 39 — demo.py

The walkthrough entry point for the Human-in-the-Loop Research Agent.

Modes (offline ones need no key — set FAKE_LLM=1 for canned replies):

    python demo.py --graph
        Draw the workflow graph. No key, no network.

    python demo.py --selftest
        Run the offline wiring tests (interrupt, resume, persistence, store,
        time-travel). No key.

    python demo.py --hitl "your research question"
        The headline demo. Run until the plan, PAUSE, show the plan, take the
        human's approve/edit from the terminal, then resume to the report.

    python demo.py --restart "your research question"
        Prove persistence: run to the pause in "process 1", throw the objects
        away, rebuild from the SAME database file, and resume in "process 2".

    python demo.py --store
        Prove cross-thread memory: run A saves a report; a SEPARATE run B on a
        related question recalls it as prior context.

    python demo.py --time-travel "your research question"
        Run to completion, list every checkpoint, then FORK an alternate run
        from an early checkpoint with update_state.

Real runs need ANTHROPIC_API_KEY (and ideally TAVILY_API_KEY; without it,
search falls back to keyless DuckDuckGo). Without a key, set FAKE_LLM=1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Disable hosted tracing (Session 26's topic, not this one). Hard-set so it
# wins over any global env var and never prints a LangSmith 403.
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


def _new_thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def run_hitl(question: str) -> None:
    """The headline demo: pause for human approval, then resume."""
    import trace as tr
    from langgraph.types import Command
    from graph import build_graph
    from persistence import build_sqlite_checkpointer, build_store
    from state import QUALITY_THRESHOLD, new_state

    app = build_graph(checkpointer=build_sqlite_checkpointer(), store=build_store())
    cfg = _new_thread_config("hitl-demo")
    tr.banner(f"HUMAN-IN-THE-LOOP RESEARCH  -  {question}")

    # Phase 1: run until the interrupt. The graph stops inside human_review.
    interrupted = app.invoke(new_state(question), cfg)
    snap = app.get_state(cfg)
    tr.recall_line(bool(snap.values.get("prior_context")))
    tr.plan_line(snap.values.get("plan", ""))

    payload = interrupted["__interrupt__"][0].value
    tr.pause_line(payload)

    # Phase 2: take the human decision from the terminal.
    try:
        decision = input("\n   Your decision (Enter = approve, or type an edited plan): ").strip()
    except EOFError:
        decision = "approve"
    decision = decision or "approve"
    tr.resume_line(decision)

    # Phase 3: resume. The research loop now runs to a grounded report.
    final: dict = {}
    for step in app.stream(Command(resume=decision), cfg):
        for node_name, update in step.items():
            update = update or {}   # a node that returns {} streams as None
            final.update(update)
            tr.node_enter(node_name)
            if node_name == "search":
                tr.search_line(final.get("refined_query", ""),
                               len(update.get("search_results", [])),
                               len(final.get("search_results", [])))
            elif node_name == "evaluate":
                tr.gate_line(update.get("quality_score", 0.0), QUALITY_THRESHOLD,
                             update.get("quality_reason", ""))
            elif node_name == "write_report":
                tr.report_line()
            elif node_name == "save":
                tr.save_line()

    tr.final_report(final.get("report", "(no report produced)"))


def run_restart(question: str) -> None:
    """Prove the State survives a program restart."""
    import trace as tr
    from langgraph.types import Command
    from graph import build_graph
    from persistence import build_sqlite_checkpointer, build_store
    from state import new_state

    import tempfile
    db_path = os.path.join(tempfile.mkdtemp(), "restart_demo.db")
    cfg = _new_thread_config("restart-demo")
    tr.banner("PERSISTENCE — STATE SURVIVES A RESTART")

    print("\n[process 1] starting a run, pausing at human review...")
    cp1 = build_sqlite_checkpointer(db_path)
    app1 = build_graph(checkpointer=cp1, store=build_store())
    app1.invoke(new_state(question), cfg)
    print(f"[process 1] paused. proposed plan saved to {db_path}")
    print("[process 1] >>> closing the db + deleting all objects (simulating program exit) <<<")
    cp1.conn.close()
    del app1, cp1

    print("\n[process 2] fresh program. Re-opening the SAME database file...")
    app2 = build_graph(checkpointer=build_sqlite_checkpointer(db_path), store=build_store())
    snap = app2.get_state(cfg)
    print(f"[process 2] reloaded. graph is paused at: {snap.next}")
    tr.plan_line(snap.values.get("plan", ""))
    final = app2.invoke(Command(resume="approve"), cfg)
    print(f"\n[process 2] resumed and finished. report length: {len(final.get('report',''))} chars")
    tr.final_report(final.get("report", "")[:400])


def run_store() -> None:
    """Prove cross-thread long-term memory via the Store."""
    import trace as tr
    from langgraph.types import Command
    from graph import build_graph
    from persistence import REPORTS_NAMESPACE, build_sqlite_checkpointer, build_store
    from state import new_state

    store = build_store()  # ONE store, shared across both runs
    app = build_graph(checkpointer=build_sqlite_checkpointer(), store=store)
    tr.banner("LONG-TERM MEMORY — THE STORE IS CROSS-THREAD")

    print("\n[run A] researching 'what is vector database indexing' ...")
    cfg_a = _new_thread_config("store-A")
    app.invoke(new_state("what is vector database indexing"), cfg_a)
    app.invoke(Command(resume="approve"), cfg_a)
    print(f"[run A] saved. store now holds {len(store.search(REPORTS_NAMESPACE, limit=10))} report(s).")

    print("\n[run B] a SEPARATE run: 'how does vector database indexing scale' ...")
    cfg_b = _new_thread_config("store-B")
    app.invoke(new_state("how does vector database indexing scale"), cfg_b)
    prior = app.get_state(cfg_b).values.get("prior_context", "")
    if prior:
        print("[run B] recall_node surfaced knowledge from run A (the checkpointer could NOT do this):")
        tr.plan_line(prior)
    else:
        print("[run B] no prior context recalled.")


def run_time_travel(question: str) -> None:
    """List every checkpoint, then fork an alternate run from an early one."""
    import trace as tr
    from langgraph.types import Command
    from graph import build_graph
    from persistence import build_sqlite_checkpointer, build_store
    from state import new_state

    app = build_graph(checkpointer=build_sqlite_checkpointer(), store=build_store())
    cfg = _new_thread_config("tt-demo")
    tr.banner("TIME-TRAVEL — REWIND AND FORK A RUN")

    app.invoke(new_state(question), cfg)
    final = app.invoke(Command(resume="approve"), cfg)
    print(f"\noriginal run finished. report length: {len(final.get('report',''))} chars")

    history = list(app.get_state_history(cfg))
    tr.timetravel_line(f"this run produced {len(history)} checkpoints:")
    for h in history:
        nxt = h.next or ("END",)
        cid = h.config["configurable"]["checkpoint_id"][:8]
        print(f"     checkpoint {cid}  next={nxt}")

    early = history[-1]
    forked_cfg = app.update_state(early.config, {"question": question + " (FORKED variant)"})
    forked_q = app.get_state(forked_cfg).values.get("question")
    tr.timetravel_line(f"forked an alternate branch from the first checkpoint: question is now {forked_q!r}")


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
    if args[0] == "--selftest":
        import selftest
        selftest.run()
        return
    if args[0] == "--restart":
        run_restart(" ".join(args[1:]) or "what is agent memory")
        return
    if args[0] == "--store":
        run_store()
        return
    if args[0] == "--time-travel":
        run_time_travel(" ".join(args[1:]) or "what is chunking")
        return
    if args[0] == "--hitl":
        run_hitl(" ".join(args[1:]) or "what is retrieval-augmented generation")
        return

    # default: treat the whole arg string as the question for the HITL demo
    run_hitl(" ".join(args))


if __name__ == "__main__":
    main()
