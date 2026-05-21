"""
Session 29 — retry.py

The production pattern this session introduces:

    Retry logic for agent tool calls — if a tool fails, the agent
    retries the call (with backoff) rather than crashing.

Why does this exist as its own module?
  - Every agent in Phase 5 will call tools, and every tool will sometimes
    fail (network blip, rate limit, malformed input). Putting the retry
    helper in one file means every agent gets the same retry semantics
    for free.
  - Production agents have observability AND retry. You need both. We
    pair the retry with a trace event so the human reading the log later
    can see exactly which attempt produced which result.

Why exponential backoff:
  - Most transient failures (rate limits, brief 5xx errors) recover on the
    next request. A fixed sleep wastes time; a no-sleep retry hammers the
    upstream. Exponential backoff is the production default.

This is NOT the same as LLM-call retry — that lives in agent.py. This file
is ONLY for tool calls.
"""

from __future__ import annotations

import time
from typing import Callable


DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_INITIAL_BACKOFF_SECONDS = 0.5


def run_tool_with_retry(
    tool_callable: Callable[[str], str],
    tool_input: str,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    on_attempt: Callable[[int, str], None] | None = None,
) -> str:
    """Execute a tool, retrying on raised exceptions with exponential backoff.

    A returned string that starts with "ERROR:" is NOT a retry trigger — the
    tool itself decided that's the answer. We only retry on raised exceptions
    (network failures, tool crashes, etc.). That distinction matters: it lets
    the LLM see the structured error and decide its own recovery.
    """
    backoff_seconds = initial_backoff_seconds
    last_exception: Exception | None = None

    for attempt_number in range(1, max_attempts + 1):
        if on_attempt is not None:
            on_attempt(attempt_number, tool_input)

        try:
            return tool_callable(tool_input)
        except Exception as exc:                            # noqa: BLE001
            last_exception = exc
            if attempt_number == max_attempts:
                break
            time.sleep(backoff_seconds)
            backoff_seconds *= 2  # exponential

    # If we drop out of the loop, every attempt raised. Surface a structured
    # error string so the LLM can read it on the next loop turn — the agent
    # does not crash.
    return f"ERROR: tool raised on every attempt — last exception: {last_exception!r}"
