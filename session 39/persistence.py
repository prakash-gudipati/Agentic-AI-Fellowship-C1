"""
Session 39 — persistence.py

THE FIRST HEADLINE IDEA: persistence. This module builds the two memory
substrates the graph plugs into. They are DIFFERENT things and beginners
constantly conflate them, so the whole point of this file is to keep them apart.

  CHECKPOINTER  (per-thread, short-term)
      Saves a snapshot of the State after EVERY node, tagged by thread_id.
      It is what makes a run pausable and resumable: pull up thread "abc",
      and you get back exactly the State as it was when you left it — mid-loop,
      paused at a human review, whatever. One thread = one conversation.
      Analogy: the autosave in a video game. Reload the save, resume the level.

  STORE  (cross-thread, long-term)
      A key-value memory that OUTLIVES any single run and is SHARED across
      threads. Thread A can write a fact; thread B (a totally separate run,
      maybe next week) can read it. The checkpointer never crosses threads;
      the Store is the thing that does.
      Analogy: the game's shared save file of unlocked achievements — visible
      no matter which save slot (thread) you load.

We default the checkpointer to a SQLite FILE so state survives a program
restart (the portfolio requirement). Set S39_DB_PATH to relocate the file off
a restrictive/networked filesystem. For the offline selftest we use the
in-memory variants — fast, disposable, no file locking on the mount.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore

_DEFAULT_DB = Path(__file__).with_name("research_agent.db")


def build_sqlite_checkpointer(db_path: str | None = None) -> SqliteSaver:
    """A durable, file-backed checkpointer. State survives a process restart.

    The SQLite connection uses check_same_thread=False because LangGraph may
    touch it from a worker thread. journal_mode=MEMORY + synchronous=NORMAL
    avoid POSIX file-lock failures on FUSE / networked mounts (the same fix
    S32's memory_store.py used).
    """
    path = db_path or os.environ.get("S39_DB_PATH") or str(_DEFAULT_DB)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=MEMORY;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return SqliteSaver(conn)


def build_memory_checkpointer() -> InMemorySaver:
    """A disposable in-RAM checkpointer. Used by the offline selftest only —
    nothing survives the process, which is exactly what a unit test wants."""
    return InMemorySaver()


def build_store() -> InMemoryStore:
    """The cross-thread long-term Store.

    We use the in-memory Store for teaching: it is the same API as the
    production Postgres-backed store, minus the deployment. The README points
    at langgraph's persistent store implementations for production.
    """
    return InMemoryStore()


# The namespace the recall/save nodes use inside the Store. A namespace is just
# a tuple key prefix — think of it as a folder. We keep all finished reports
# under ("research", "reports").
REPORTS_NAMESPACE = ("research", "reports")
