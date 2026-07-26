"""
Session 30 — action_log.py

PROD PATTERN introduced this session:

    Agent action logging — every tool call leaves a single, structured
    JSON line in action_log.jsonl recording what the agent asked for and
    what it got back.

Why does this exist as its own module?
  - Pretty terminal output is for the human running the demo. JSONL is
    for the SRE on call at 2am, the data scientist building evals, the
    PM trying to understand why a customer's run failed yesterday. They
    cannot scroll back through coloured terminal output; they need a file
    they can grep, jq, and load into pandas.
  - One module owns the schema of that file. If the schema changes, every
    consumer downstream sees the change in ONE place.

The schema (every line in action_log.jsonl):

  {
    "ts":            "2026-05-21T14:03:11.481Z",
    "run_id":        "run-20260521-140311-abc123",
    "turn":          2,
    "tool_name":     "web_search",
    "tool_args":     {"query": "population of India 2024"},
    "tool_output":   "UN Population Division: India's estimated population ...",
    "duration_ms":   142,
    "attempts":      1,
    "status":        "ok"        // "ok" | "error" | "schema_violation"
  }

A few design choices worth seeing:
  - run_id binds every line from one agent run together. Trivial to filter
    `jq 'select(.run_id == "run-...")'` and reconstruct a single run.
  - tool_output is stored as a STRING — even if the tool returned JSON.
    We never lie about what the LLM actually saw.
  - duration_ms is in milliseconds, integer, never fractional. Aggregate-
    friendly.
  - status is a small, finite vocabulary. Easy to chart.

Production patterns introduced:
  - Structured JSONL logging                                       (S30 new)
  - Stable schema, single owner                                    (S30 new)
  - run_id correlation across multiple lines                       (S30 new)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ── Where the log lives ─────────────────────────────────────────────────────
# Path is relative to the file that runs the agent (typically demo.py).
# In a real product this would be configurable; for the session we just
# point at the working directory so students can `cat` it after a run.
DEFAULT_LOG_PATH = Path("action_log.jsonl")


# ── Status vocabulary — keep this list short on purpose ─────────────────────
STATUS_OK = "ok"
STATUS_ERROR = "error"                # tool ran but returned ERROR: ...
STATUS_SCHEMA_VIOLATION = "schema_violation"   # provider rejected the call


def new_run_id() -> str:
    """Stable, sortable, locally unique. Date-prefixed so `ls` shows runs in order."""
    short = uuid.uuid4().hex[:6]
    return f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{short}"


@dataclass
class ActionRecord:
    """One tool call. One JSON line. Everything an SRE needs to debug a single step."""
    ts: str
    run_id: str
    turn: int
    tool_name: str
    tool_args: Dict[str, Any]
    tool_output: str
    duration_ms: int
    attempts: int
    status: str
    extra: Dict[str, Any] = field(default_factory=dict)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class ActionLog:
    """A small append-only writer for one run.

    Usage:
        log = ActionLog(run_id="run-...")
        with log.measure(turn=2, tool_name="web_search", tool_args={...}) as ctx:
            result = tool.run({...})
            ctx.finish(result, attempts=1, status="ok")
    """

    def __init__(
        self,
        run_id: str,
        log_path: Path | str = DEFAULT_LOG_PATH,
    ) -> None:
        self.run_id = run_id
        self.log_path = Path(log_path)
        # Make sure the directory exists. We never auto-delete a log file —
        # logs are append-only by contract.
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: ActionRecord) -> None:
        """Append one ActionRecord to the JSONL file."""
        line = json.dumps(asdict(record), ensure_ascii=False, default=str)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def measure(
        self,
        *,
        turn: int,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> "_MeasureContext":
        """Return a context manager that times one tool call and writes the line on exit."""
        return _MeasureContext(self, turn=turn, tool_name=tool_name, tool_args=tool_args)


class _MeasureContext:
    """Internal — built by ActionLog.measure(). Don't instantiate directly."""

    def __init__(
        self,
        parent: ActionLog,
        *,
        turn: int,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> None:
        self.parent = parent
        self.turn = turn
        self.tool_name = tool_name
        self.tool_args = tool_args
        self._start_ns = 0
        self._tool_output: Optional[str] = None
        self._attempts: int = 1
        self._status: str = STATUS_OK
        self._extra: Dict[str, Any] = {}

    def __enter__(self) -> "_MeasureContext":
        self._start_ns = time.perf_counter_ns()
        return self

    def finish(
        self,
        tool_output: str,
        *,
        attempts: int = 1,
        status: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record what the tool returned. Call this from inside the `with` block."""
        self._tool_output = tool_output
        self._attempts = attempts
        # Auto-detect status if the caller did not specify one.
        if status is None:
            status = STATUS_ERROR if tool_output.startswith("ERROR:") else STATUS_OK
        self._status = status
        if extra:
            self._extra.update(extra)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = int((time.perf_counter_ns() - self._start_ns) / 1_000_000)

        # If the body raised, capture that as a schema_violation-style line so
        # we never silently lose a tool call.
        if exc_type is not None:
            output = f"PYTHON-EXCEPTION: {exc_val!r}"
            status = STATUS_SCHEMA_VIOLATION
        elif self._tool_output is None:
            # Caller forgot to call finish(). Record it; don't crash.
            output = "WARNING: action_log.finish() was not called for this tool invocation."
            status = STATUS_ERROR
        else:
            output = self._tool_output
            status = self._status

        record = ActionRecord(
            ts=_utc_iso(),
            run_id=self.parent.run_id,
            turn=self.turn,
            tool_name=self.tool_name,
            tool_args=self.tool_args,
            tool_output=output,
            duration_ms=duration_ms,
            attempts=self._attempts,
            status=status,
            extra=self._extra,
        )
        self.parent.write(record)
        # Do not suppress exceptions — let them propagate.
        return False


def summarise_run(log_path: Path | str = DEFAULT_LOG_PATH, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Quick aggregate over the JSONL file. Used by demo.py at the end of a run."""
    path = Path(log_path)
    if not path.exists():
        return {"runs": 0, "tool_calls": 0}

    tool_calls = 0
    errors = 0
    schema_violations = 0
    total_ms = 0
    tool_names: Dict[str, int] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run_id and record.get("run_id") != run_id:
                continue
            tool_calls += 1
            total_ms += int(record.get("duration_ms", 0))
            status = record.get("status", "ok")
            if status == STATUS_ERROR:
                errors += 1
            if status == STATUS_SCHEMA_VIOLATION:
                schema_violations += 1
            name = record.get("tool_name", "<unknown>")
            tool_names[name] = tool_names.get(name, 0) + 1

    return {
        "tool_calls": tool_calls,
        "errors": errors,
        "schema_violations": schema_violations,
        "total_duration_ms": total_ms,
        "by_tool": tool_names,
    }
