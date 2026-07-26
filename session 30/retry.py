"""
Session 30 - retry.py

Exponential-backoff retry for tool calls.

v3 update (S30 meaty rebuild): now respects the `idempotent` flag on the
Tool dataclass. If a tool is marked non-idempotent (charges a card, sends
an email, books a flight), this helper will NOT re-run it after a failure.
It returns the error to the model directly so the model can decide.

The argument: re-running a non-idempotent tool is often worse than failing.
Charging a card twice is worse than charging it zero times. Email sent
twice is worse than not sent. The retry helper does not get to make that
call - the tool author does, via the idempotent flag.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Tuple


DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_INITIAL_BACKOFF_SECONDS = 0.5


def run_tool_with_retry(
    tool_callable: Callable[[Dict[str, Any]], str],
    tool_args: Dict[str, Any],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    idempotent: bool = True,
) -> Tuple[str, int]:
    """Execute a tool, retrying on raised exceptions.

    Parameters
    ----------
    idempotent : bool, default True
        When False, the helper makes AT MOST ONE attempt. If that attempt
        raises, the error is surfaced as a structured string. The model
        gets to see it and decide what to do next - we do not silently
        re-execute a side-effectful tool.

    Returns
    -------
    (observation_string, attempts_used)

    A returned string starting with 'ERROR:' is NOT a retry trigger.
    """
    # Non-idempotent tools: ONE attempt only. No backoff loop.
    if not idempotent:
        try:
            return tool_callable(tool_args), 1
        except Exception as exc:                            # noqa: BLE001
            return (
                f"ERROR[permanent]: non-idempotent tool failed on first attempt: {exc!r}. "
                f"NOT retried - tool author marked this as having side effects.",
                1,
            )

    # Idempotent: normal retry loop with exponential backoff.
    backoff_seconds = initial_backoff_seconds
    last_exception: Exception | None = None

    for attempt_number in range(1, max_attempts + 1):
        try:
            return tool_callable(tool_args), attempt_number
        except Exception as exc:                            # noqa: BLE001
            last_exception = exc
            if attempt_number == max_attempts:
                break
            time.sleep(backoff_seconds)
            backoff_seconds *= 2

    return (
        f"ERROR[transient]: tool raised on every attempt - last exception: {last_exception!r}",
        max_attempts,
    )
