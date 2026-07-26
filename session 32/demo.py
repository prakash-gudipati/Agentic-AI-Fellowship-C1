"""
Session 32 — demo.py

Six demos that exercise every memory layer in turn.

Usage:
    python demo.py 1     # The amnesia baseline (no memory at all)
    python demo.py 2     # Working memory only (verbatim sliding window)
    python demo.py 3     # Summarisation kicks in when the budget overflows
    python demo.py 4     # Semantic recall pulls back an old fact
    python demo.py 5     # Save / load: session state as JSON artifact
    python demo.py 6     # PRODUCTION SUBSTRATE — SQLite-backed memory
                         # across two processes (close + reopen the DB).

By default these demos make REAL Anthropic API calls. Drop your
ANTHROPIC_API_KEY in a .env file next to this script. To run offline
against the canned fake LLM, prefix with FAKE_LLM=1:

    FAKE_LLM=1 python demo.py 6
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ----------------------------------------------------------------------------
# Tiny .env loader (no python-dotenv dependency)
# ----------------------------------------------------------------------------


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).parent / ".env")


from agent import MemoryAgent
from eviction import HybridPolicy, RecencyPolicy
from llm_client import LLMClient
from memory_store import MemoryStore
from session_state import SessionStore
from tools import REGISTRY
from trace_logger import (
    print_assistant,
    print_block,
    print_context_ledger,
    print_eviction,
    print_section,
    print_semantic_hits,
    print_summary,
    print_user,
    print_working_state,
)


if os.environ.get("FAKE_LLM", "") != "1" and not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit(
        "\nERROR: ANTHROPIC_API_KEY is not set in your environment.\n"
        "       Either:\n"
        "         1) export ANTHROPIC_API_KEY=sk-ant-...   (real calls)\n"
        "         2) export FAKE_LLM=1                     (offline canned mode)\n"
        "       then re-run.\n"
    )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _walk(agent: MemoryAgent, user_message: str) -> None:
    """Print one labelled turn through the memory pipeline."""

    print_section("turn: user message")
    print_user(user_message)

    trace = agent.chat(user_message)

    print_assistant(trace.assistant_text)
    print_working_state(
        agent.working.turns(),
        agent.working.token_count(),
        agent.working.token_budget,
    )
    print_eviction(trace.eviction_events)
    if trace.summarised or (agent.summary and agent.summary.text):
        print_summary(agent.summary)
    if trace.semantic_hits:
        print_semantic_hits(user_message, trace.semantic_hits)
    if trace.context is not None:
        print_context_ledger(
            trace.context.breakdown, trace.context.total_tokens()
        )


# ----------------------------------------------------------------------------
# Demo 1 — amnesia baseline
# ----------------------------------------------------------------------------


def demo_1_amnesia() -> None:
    print_block(
        "DEMO 1 — Amnesia baseline (no working memory)",
        "The agent has WORKING_BUDGET=1 forced eviction. After every turn "
        "it cannot recall what was said before. This is the shape every "
        "fresh API call has by default — no memory.",
    )
    agent = MemoryAgent(
        working_budget=1,
        working_policy=RecencyPolicy(min_keep=0),
        enable_summary=False,
        enable_semantic=False,
    )
    _walk(agent, "My name is Aanya and I am from Bengaluru.")
    _walk(agent, "I work on a CLI productivity tool.")
    _walk(agent, "What is my name?")
    _walk(agent, "Remind me my favourite city is.")


# ----------------------------------------------------------------------------
# Demo 2 — working memory only
# ----------------------------------------------------------------------------


def demo_2_working_memory_only() -> None:
    print_block(
        "DEMO 2 — Working memory only (verbatim sliding window)",
        "All four memory layers are present in code but summary and "
        "semantic are switched OFF. Working memory alone carries the "
        "recent past verbatim. Watch the budget tick up.",
    )
    agent = MemoryAgent(
        working_budget=400,
        working_policy=HybridPolicy(k=4, min_keep=2),
        enable_summary=False,
        enable_semantic=False,
    )
    _walk(agent, "My name is Aanya. I am building a study scheduler.")
    _walk(agent, "I am from Bengaluru and my favourite city is Lisbon.")
    _walk(agent, "What is my name?")
    _walk(agent, "What is my favourite city?")


# ----------------------------------------------------------------------------
# Demo 3 — summarisation kicks in
# ----------------------------------------------------------------------------


def demo_3_summarisation() -> None:
    print_block(
        "DEMO 3 — Conversation Summarisation (PROD PATTERN)",
        "Token budget is forced low so eviction triggers fast. As old "
        "turns leave, the summariser folds them into a rolling summary "
        "that travels alongside the verbatim working buffer.",
    )
    agent = MemoryAgent(
        working_budget=120,
        working_policy=HybridPolicy(k=2, min_keep=2),
        enable_summary=True,
        enable_semantic=False,
    )

    facts = [
        "My name is Aanya, I am from Bengaluru, my favourite city is Lisbon.",
        "I am building a study scheduler. The project is called PrepDeck.",
        "I prefer SQLite over Postgres for the prototype.",
        "My target launch month is October 2026.",
        "I want a CLI front-end first and a web app later.",
        "I plan to charge in INR with a free tier.",
        "What is my name?",
        "What is my project called?",
        "What month do I want to launch?",
    ]
    for f in facts:
        _walk(agent, f)


# ----------------------------------------------------------------------------
# Demo 4 — semantic recall pulls back an OLD fact
# ----------------------------------------------------------------------------


def demo_4_semantic_recall() -> None:
    print_block(
        "DEMO 4 — Semantic Memory via Vector Recall (PROD PATTERN)",
        "Plant a Fact One. Spend several turns on unrelated chat so the "
        "fact is evicted out of working memory. Then ask about it again. "
        "The agent must pull it back from the vector store.",
    )
    agent = MemoryAgent(
        working_budget=80,
        working_policy=HybridPolicy(k=2, min_keep=2),
        enable_summary=False,
        enable_semantic=True,
        semantic_min_score=-1.0,
    )

    _walk(agent, "Fact One: the launch month is October 2026.")
    _walk(agent, "Let's talk about something else for a bit.")
    _walk(agent, "The weather in Bengaluru today is humid and warm.")
    _walk(agent, "Bengaluru traffic peaks between 6 and 9 PM most days.")
    _walk(agent, "I had filter coffee with biscuits for breakfast today.")
    _walk(agent, "What is the launch month?")


# ----------------------------------------------------------------------------
# Demo 5 — session state as JSON artifact
# ----------------------------------------------------------------------------


def demo_5_session_state_artifact() -> None:
    print_block(
        "DEMO 5 — Session State as Artifact (PROD PATTERN, JSON path)",
        "Run a short session, save the snapshot to disk, hydrate a NEW "
        "agent from the same JSON file, confirm it still answers from "
        "carried-over memory.",
    )

    store = SessionStore(root=str(Path(__file__).parent / "_sessions"))

    agent_a = MemoryAgent(working_budget=300)
    _walk(agent_a, "My name is Aanya. I'm building PrepDeck for JEE aspirants.")
    _walk(agent_a, "My favourite city is Lisbon and my target launch is October 2026.")
    _walk(agent_a, "I'll see you tomorrow.")

    snapshot = agent_a.snapshot()
    saved_at = store.save(snapshot)
    print(
        f"\nSaved session state to: {saved_at}\n"
        f"  working_turns      = {len(snapshot.working_turns)}\n"
        f"  summary_notes      = {len(snapshot.summary_notes)}\n"
        f"  semantic_turn_ids  = {len(snapshot.semantic_turn_ids)}\n"
        f"  eviction_log       = {len(snapshot.eviction_log)} event(s)"
    )

    print_section("rehydrating a fresh agent from the JSON file")
    loaded = store.load(snapshot.session_id)
    agent_b = MemoryAgent.from_snapshot(loaded, client=LLMClient())

    _walk(agent_b, "Remind me my name and my project name.")
    _walk(agent_b, "What is my favourite city?")


# ----------------------------------------------------------------------------
# Demo 6 — PRODUCTION SUBSTRATE — SQLite-backed memory across processes
# ----------------------------------------------------------------------------


def demo_6_sqlite_substrate() -> None:
    print_block(
        "DEMO 6 — Production substrate (SQLite-backed memory)",
        "Same four PROD PATTERNS, but every memory mutation is mirrored "
        "to a SQLite file. Then we CLOSE the store (simulating the "
        "Python process exiting), open a fresh MemoryStore against the "
        "same file, and rehydrate an agent that answers from the DB.",
    )

    # Default DB path sits next to the code; override with S32_DB_PATH=...
    # for filesystems where SQLite cannot hold a file lock (e.g. some
    # virtualised classroom mounts or FUSE-mounted Windows shares).
    db_path = Path(os.environ.get(
        "S32_DB_PATH",
        str(Path(__file__).parent / "_sessions" / "memory.db"),
    ))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Start clean so the demo is reproducible. Open the file (creates
    # it if needed), clear every table, then close. This is more
    # production-realistic than unlinking — you almost never delete
    # a live DB file; you truncate or DELETE the rows.
    _reset = MemoryStore.open(db_path)
    with _reset.conn:
        for tbl in ("eviction_log", "vectors", "summaries", "turns", "sessions"):
            _reset.conn.execute(f"DELETE FROM {tbl}")
    _reset.close()

    # ---- Phase A: write ----------------------------------------------------
    print_section("Phase A — write (Process A)")
    store_a = MemoryStore.open(db_path)
    print(f"  DB file: {db_path}")
    print(f"  sessions in store: {store_a.list_sessions()}")

    # 120-tok budget guarantees eviction fires by turn 3 so every
    # PROD PATTERN (summary, eviction log, semantic vectors) lands rows
    # in the DB before Process A closes.
    agent_a = MemoryAgent(
        working_budget=120,
        working_policy=HybridPolicy(k=2, min_keep=2),
        enable_summary=True,
        enable_semantic=True,
        semantic_min_score=-1.0,
        store=store_a,
    )
    session_id = agent_a.session_id
    print(f"  new session_id: {session_id}")

    # Plant identity facts, then a few overflow turns so summarisation +
    # eviction fire and we get real DB rows in every table.
    for msg in [
        "My name is Aanya. I'm building PrepDeck for JEE aspirants.",
        "My favourite city is Lisbon.",
        "My target launch month is October 2026.",
        "I prefer SQLite over Postgres for the prototype.",
        "I plan to charge in INR with a free tier.",
        "Today is humid in Bengaluru.",
    ]:
        _walk(agent_a, msg)

    # Inspect the DB directly so the room can see real rows landed.
    n_turns_total = store_a.conn.execute(
        "SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    n_turns_working = store_a.conn.execute(
        "SELECT COUNT(*) FROM turns WHERE session_id = ? AND in_working = 1",
        (session_id,),
    ).fetchone()[0]
    n_vectors = store_a.conn.execute(
        "SELECT COUNT(*) FROM vectors WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    n_evicted_log = store_a.conn.execute(
        "SELECT COUNT(*) FROM eviction_log WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    has_summary = store_a.conn.execute(
        "SELECT 1 FROM summaries WHERE session_id = ?", (session_id,)
    ).fetchone() is not None

    print()
    print("  --- DB inspection (Process A) ---")
    print(f"    turns total      : {n_turns_total}")
    print(f"    turns in_working : {n_turns_working}")
    print(f"    vectors          : {n_vectors}")
    print(f"    eviction_log     : {n_evicted_log} event(s)")
    print(f"    summary row?     : {has_summary}")

    store_a.close()
    del agent_a, store_a
    print("\n  Process A closed. Python objects are gone. DB file remains on disk.")

    # ---- Phase B: read in a fresh "process" --------------------------------
    print_section("Phase B — read (Process B, fresh MemoryStore handle)")
    store_b = MemoryStore.open(db_path)
    print(f"  sessions visible : {store_b.list_sessions()}")

    agent_b = MemoryAgent.from_store(
        store_b,
        session_id=session_id,
        client=LLMClient(),
        working_budget=120,
        working_policy=HybridPolicy(k=2, min_keep=2),
        enable_summary=True,
        enable_semantic=True,
        semantic_min_score=-1.0,
    )
    print(
        f"  rehydrated working turns : {agent_b.working.turn_count()}\n"
        f"  rehydrated summary?      : "
        f"{'yes' if agent_b.summary else 'no'}\n"
        f"  rehydrated semantic dim  : {agent_b.semantic.size()} vector(s)\n"
        f"  rehydrated eviction log  : {len(agent_b.eviction_log)} event(s)"
    )

    # Ask the agent questions whose answers MUST come from the DB —
    # the agent has no Python state shared with Process A.
    _walk(agent_b, "Remind me my name and my project name.")
    _walk(agent_b, "What is my favourite city?")
    _walk(agent_b, "Remind me about the launch month I planted earlier.")
    store_b.close()

    print(
        "\nNote: every answer above was reconstructed from SQL rows in "
        f"{db_path.name}. No Python state crossed the boundary. "
        "This is the same shape Redis + Postgres takes in production — "
        "just compressed into one file for teaching."
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


DEMOS = {
    "1": demo_1_amnesia,
    "2": demo_2_working_memory_only,
    "3": demo_3_summarisation,
    "4": demo_4_semantic_recall,
    "5": demo_5_session_state_artifact,
    "6": demo_6_sqlite_substrate,
}


def _usage() -> None:
    print(__doc__)
    print(f"\nAvailable tools in this build: {', '.join(REGISTRY.keys())}")


def main(argv: list) -> int:
    if len(argv) < 2 or argv[1] not in DEMOS:
        _usage()
        return 0
    DEMOS[argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
