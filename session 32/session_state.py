"""
Session 32 — session_state.py

PROD PATTERN — Session State as Artifact.

Every memory layer the agent carries — working buffer, rolling summary,
semantic index, eviction log — must be serialisable to ONE JSON file at
the end of a run and re-hydratable from that same JSON at the start of
the next one. That file is the agent's "artifact".

Why a single artifact and not four?
  - In production you do not get to control whether the worker that ran
    the first session is the same worker that will run the next session.
  - You also do not get to control whether the user resumes 10 seconds
    later or 10 days later.
  - One file with a clear schema is the lowest-friction handoff.

Callbacks:
  - S31 introduced "Plan as Artifact". This is the natural extension of
    that pattern from a single plan to the entire memory state.
  - S26 (observability) showed that the act of writing structured logs
    pays off the moment something goes wrong. Same logic here.

Public surface:
  - SessionStore.save(snapshot, path)        — atomic write to disk
  - SessionStore.load(path)                  — rehydrate snapshot
  - SessionStore.list(directory)             — find session files
  - build_snapshot(...) helper for agent.py
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional

from memory_types import (
    EvictionEvent,
    SessionStateSnapshot,
    SummaryNote,
    Turn,
)


# ----------------------------------------------------------------------------
# SessionStore
# ----------------------------------------------------------------------------


class SessionStore:
    """
    Tiny on-disk store for SessionStateSnapshot.

    File format is JSON, file name is session_<session_id>.json. We do an
    atomic write (write to a tempfile, then os.replace) so a crash during
    write cannot leave a partial JSON behind.

    Production code reaches for a real KV store (Redis, DynamoDB, Postgres
    JSONB). The pattern is identical — only the persistence layer changes.
    """

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = Path(root) if root else Path.cwd() / "sessions"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, snapshot: SessionStateSnapshot) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"session_{snapshot.session_id}.json"

        payload = snapshot.to_json()

        # Atomic write — tempfile in the same dir so the rename is on the
        # same filesystem (otherwise os.replace can fall back to copy).
        fd, tmp_path = tempfile.mkstemp(
            prefix=".session_", suffix=".json", dir=str(self.root)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, target)
        except Exception:
            # Clean up the temp file if we never made it to the rename.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return target

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, session_id: str) -> SessionStateSnapshot:
        path = self.root / f"session_{session_id}.json"
        raw = path.read_text(encoding="utf-8")
        return SessionStateSnapshot.from_json(raw)

    def load_path(self, path: Path) -> SessionStateSnapshot:
        raw = path.read_text(encoding="utf-8")
        return SessionStateSnapshot.from_json(raw)

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list(self) -> List[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("session_*.json"))


# ----------------------------------------------------------------------------
# build_snapshot — the agent-side helper
# ----------------------------------------------------------------------------


def build_snapshot(
    session_id: str,
    working_turns: List[Turn],
    summary_notes: List[SummaryNote],
    semantic_turn_ids: List[str],
    eviction_log: List[EvictionEvent],
    metadata: Optional[dict] = None,
) -> SessionStateSnapshot:
    """
    Convenience constructor used by agent.py at the end of each session.

    Kept separate from SessionStateSnapshot so agent.py never has to
    fiddle with the dataclass directly.
    """

    return SessionStateSnapshot(
        session_id=session_id,
        created_at=time.time(),
        working_turns=list(working_turns),
        summary_notes=list(summary_notes),
        semantic_turn_ids=list(semantic_turn_ids),
        eviction_log=list(eviction_log),
        metadata=dict(metadata or {}),
    )


# ----------------------------------------------------------------------------
# new_session_id — small helper so all four entry points agree on format
# ----------------------------------------------------------------------------


def new_session_id() -> str:
    """8-char hex id. Short enough for filenames, wide enough not to collide."""
    return uuid.uuid4().hex[:8]
