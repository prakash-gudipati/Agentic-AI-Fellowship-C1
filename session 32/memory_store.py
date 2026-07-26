"""
Session 32 — memory_store.py

PRODUCTION SUBSTRATE — SQLite-backed memory.

The other memory modules (working_memory, summariser, semantic_memory) were
all in-process. Lists, dicts, dataclasses living in Python RAM. That's the
right teaching shape — you see the patterns without the noise of a DB.

In production you swap RAM for something durable. The cheapest production
swap is one SQLite file. That is what this module is.

What's persisted:

  sessions       — one row per agent session
  turns          — every Turn the agent has ever appended; in_working flag
                   marks turns still inside the verbatim buffer
  summaries      — one row per session; replaced (UPSERTed) on every change
  vectors        — one row per evicted Turn, vector stored as a JSON array
  eviction_log   — one row per eviction event for the trace

Semantic recall:
  Cosine similarity is computed in Python over candidates loaded from the
  vectors table. This is honest about the trade-off:
    works    — up to ~10,000 vectors per session, ~50 ms query time
    breaks   — past that, you want sqlite-vec, ChromaDB, pgvector, or a
               dedicated vector store (Pinecone / Weaviate / Qdrant).

PROD PATTERN bridge:
  This module DOES NOT INVENT new patterns. It just gives the existing four
  PROD PATTERNS a durable substrate.
    - Conversation Summarisation        → summaries table
    - Memory Eviction Policy            → eviction_log table
    - Semantic Memory via Vector Recall → vectors table
    - Session State as Artifact         → the whole DB IS the artifact

Public API (kept narrow):
  - MemoryStore.open(db_path)
  - MemoryStore.close()
  - MemoryStore.upsert_session(session_id, metadata=None)
  - MemoryStore.list_sessions() -> list[str]
  - MemoryStore.append_turn(session_id, turn)
  - MemoryStore.mark_evicted(session_id, turn_id, policy, reason)
  - MemoryStore.working_turns(session_id) -> list[Turn]
  - MemoryStore.upsert_summary(session_id, summary)
  - MemoryStore.get_summary(session_id) -> Optional[SummaryNote]
  - MemoryStore.add_vector(turn_id, session_id, vector)
  - MemoryStore.session_vectors(session_id) -> list[(Turn, vector)]
  - MemoryStore.recall(session_id, query_vector, k, min_score) -> list[SemanticHit]
  - MemoryStore.eviction_log(session_id) -> list[EvictionEvent]
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memory_types import (
    EvictionEvent,
    SemanticHit,
    SummaryNote,
    Turn,
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    metadata    TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id         TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    token_estimate  INTEGER NOT NULL,
    created_at      REAL NOT NULL,
    in_working      INTEGER NOT NULL DEFAULT 1,
    metadata        TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_turns_session   ON turns(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_turns_working   ON turns(session_id, in_working);

CREATE TABLE IF NOT EXISTS summaries (
    session_id        TEXT PRIMARY KEY,
    text              TEXT NOT NULL,
    covers_turn_ids   TEXT NOT NULL,
    token_estimate    INTEGER NOT NULL,
    updated_at        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS vectors (
    turn_id      TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    vector       TEXT NOT NULL,
    FOREIGN KEY (turn_id) REFERENCES turns(turn_id)
);
CREATE INDEX IF NOT EXISTS idx_vectors_session ON vectors(session_id);

CREATE TABLE IF NOT EXISTS eviction_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn_id     TEXT NOT NULL,
    policy      TEXT NOT NULL,
    reason      TEXT NOT NULL,
    at          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eviction_session ON eviction_log(session_id, at);
"""


class MemoryStore:
    """
    Thin wrapper around a SQLite connection.

    Constructor parameter db_path = ":memory:" gives an in-memory store
    for tests. Pass any filesystem path for durable storage.

    Connection is opened with autocommit-style behaviour — every public
    method runs inside its own short transaction. This is the simplest
    correct behaviour for a teaching codebase. Production code would
    batch where possible.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        # check_same_thread=False so the demos can simulate "fresh process"
        # by opening a new MemoryStore instance against the same file.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        # FUSE / network / Windows-mount filesystems often choke on
        # SQLite's POSIX locking. journal_mode=MEMORY keeps the
        # rollback journal in RAM (the main DB still writes to disk),
        # and synchronous=NORMAL avoids fsync stalls on slow mounts.
        self.conn.execute("PRAGMA journal_mode = MEMORY")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)

    @classmethod
    def open(cls, db_path: str | Path) -> "MemoryStore":
        return cls(str(db_path))

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def upsert_session(
        self, session_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        now = time.time()
        meta_json = json.dumps(metadata or {})
        with self.conn:
            self.conn.execute(
                "INSERT INTO sessions(session_id, created_at, updated_at, metadata) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "  updated_at = excluded.updated_at, "
                "  metadata   = excluded.metadata",
                (session_id, now, now, meta_json),
            )

    def list_sessions(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT session_id FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [r["session_id"] for r in rows]

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------

    def append_turn(self, session_id: str, turn: Turn) -> None:
        """Insert a Turn with in_working=1. Idempotent on turn_id."""

        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO turns "
                "(turn_id, session_id, role, content, token_estimate, "
                "created_at, in_working, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    turn.turn_id,
                    session_id,
                    turn.role,
                    turn.content,
                    int(turn.token_estimate),
                    float(turn.created_at),
                    json.dumps(turn.metadata or {}),
                ),
            )
            self._touch_session(session_id)

    def mark_evicted(
        self,
        session_id: str,
        turn_id: str,
        policy: str,
        reason: str,
    ) -> None:
        """Flip in_working=0 for one turn and log the event."""

        with self.conn:
            self.conn.execute(
                "UPDATE turns SET in_working = 0 WHERE turn_id = ?", (turn_id,)
            )
            self.conn.execute(
                "INSERT INTO eviction_log "
                "(session_id, turn_id, policy, reason, at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, turn_id, policy, reason, time.time()),
            )
            self._touch_session(session_id)

    def working_turns(self, session_id: str) -> List[Turn]:
        rows = self.conn.execute(
            "SELECT * FROM turns "
            "WHERE session_id = ? AND in_working = 1 "
            "ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_turn(r) for r in rows]

    def all_turns(self, session_id: str) -> List[Turn]:
        rows = self.conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_turn(r) for r in rows]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def upsert_summary(self, session_id: str, summary: SummaryNote) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO summaries "
                "(session_id, text, covers_turn_ids, token_estimate, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "  text            = excluded.text, "
                "  covers_turn_ids = excluded.covers_turn_ids, "
                "  token_estimate  = excluded.token_estimate, "
                "  updated_at      = excluded.updated_at",
                (
                    session_id,
                    summary.text,
                    json.dumps(summary.covers_turn_ids),
                    int(summary.token_estimate),
                    float(summary.created_at),
                ),
            )
            self._touch_session(session_id)

    def get_summary(self, session_id: str) -> Optional[SummaryNote]:
        row = self.conn.execute(
            "SELECT * FROM summaries WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        return SummaryNote(
            text=row["text"],
            covers_turn_ids=json.loads(row["covers_turn_ids"]),
            token_estimate=int(row["token_estimate"]),
            created_at=float(row["updated_at"]),
        )

    # ------------------------------------------------------------------
    # Vectors / Semantic recall
    # ------------------------------------------------------------------

    def add_vector(
        self,
        turn_id: str,
        session_id: str,
        vector: List[float],
    ) -> None:
        """Idempotent on turn_id."""

        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO vectors (turn_id, session_id, vector) "
                "VALUES (?, ?, ?)",
                (turn_id, session_id, json.dumps(vector)),
            )

    def session_vectors(
        self, session_id: str
    ) -> List[Tuple[Turn, List[float]]]:
        """
        All (Turn, vector) pairs for a session.

        Joined against the turns table so the caller gets a real Turn
        object, not just a turn_id.
        """

        rows = self.conn.execute(
            "SELECT t.*, v.vector AS v_json "
            "FROM vectors v JOIN turns t ON t.turn_id = v.turn_id "
            "WHERE v.session_id = ?",
            (session_id,),
        ).fetchall()
        out: List[Tuple[Turn, List[float]]] = []
        for r in rows:
            turn = self._row_to_turn(r)
            vec = json.loads(r["v_json"])
            out.append((turn, vec))
        return out

    def recall(
        self,
        session_id: str,
        query_vector: List[float],
        k: int = 3,
        min_score: float = 0.15,
    ) -> List[SemanticHit]:
        """
        Top-K cosine recall over the session's vectors.

        Implementation note: we SELECT every vector for the session and
        compute cosine in Python. This is honest about the scale ceiling
        — sweet spot is ~thousands of vectors per session. Past that you
        either (a) install sqlite-vec for SQL-side similarity search, or
        (b) move the vectors out to a real vector store.

        Cosine reduces to dot product because both vectors are stored
        L2-normalised (see semantic_memory.HashEmbedder).
        """

        candidates = self.session_vectors(session_id)
        if not candidates:
            return []
        scored = [
            SemanticHit(turn=t, score=_dot(query_vector, v))
            for (t, v) in candidates
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return [h for h in scored[:k] if h.score >= min_score]

    # ------------------------------------------------------------------
    # Eviction log
    # ------------------------------------------------------------------

    def eviction_log(self, session_id: str) -> List[EvictionEvent]:
        rows = self.conn.execute(
            "SELECT turn_id, policy, reason, at FROM eviction_log "
            "WHERE session_id = ? ORDER BY at ASC",
            (session_id,),
        ).fetchall()
        return [
            EvictionEvent(
                turn_id=r["turn_id"],
                policy=r["policy"],
                reason=r["reason"],
                at=float(r["at"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _touch_session(self, session_id: str) -> None:
        """Bump updated_at; insert a session row if it does not exist yet."""

        self.conn.execute(
            "INSERT INTO sessions(session_id, created_at, updated_at, metadata) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at",
            (session_id, time.time(), time.time(), json.dumps({})),
        )

    def _row_to_turn(self, row: sqlite3.Row) -> Turn:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return Turn(
            role=row["role"],
            content=row["content"],
            token_estimate=int(row["token_estimate"]),
            created_at=float(row["created_at"]),
            turn_id=row["turn_id"],
            metadata=meta,
        )


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
