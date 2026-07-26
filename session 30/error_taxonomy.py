"""
Session 30 - error_taxonomy.py

Real tool failures don't all need the same response. This module names the
four error categories every production agent has to handle differently.

The four categories:

  TRANSIENT          - Network blip, 5xx, rate limit, timeout.
                       Recovery: retry with backoff (retry.py handles it).

  PERMANENT          - Tool itself is broken, mis-deployed, dead upstream API.
                       Recovery: skip this tool for the rest of the run.
                       Surface the failure in the action log for an SRE to triage.

  INVALID_INPUT      - The model passed args the tool can't accept, even
                       though they passed JSON Schema validation.
                       (E.g., calculator gets '1/0', or web_search gets
                       a query with disallowed chars.)
                       Recovery: return a structured error string the
                       MODEL can read and recover from on the next turn.

  TOOL_NOT_AVAILABLE - The model emitted a tool_name that doesn't exist in
                       the registry (it tried to invent a tool, or the
                       schema list shipped to it was stale).
                       Recovery: tell the model the available tools and
                       let it retry the choice.

Each category maps to a specific recovery strategy in agent.py and a
specific status in action_log.jsonl.

Production payoff: when something breaks in production at 2am, the SRE
opens action_log.jsonl, groups by error_category, and knows immediately
which category needs a human (PERMANENT, TOOL_NOT_AVAILABLE) vs which
category recovered on its own (TRANSIENT, INVALID_INPUT).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorCategory(str, Enum):
    """The four error categories every tool failure falls into."""
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    INVALID_INPUT = "invalid_input"
    TOOL_NOT_AVAILABLE = "tool_not_available"


# ── Recovery strategy per category (the agent loop reads this) ──────────────
RECOVERY = {
    ErrorCategory.TRANSIENT: {
        "retry_with_backoff": True,
        "tell_model": False,
        "skip_tool_for_rest_of_run": False,
        "log_status": "transient_recovered",
    },
    ErrorCategory.PERMANENT: {
        "retry_with_backoff": False,
        "tell_model": True,
        "skip_tool_for_rest_of_run": True,
        "log_status": "permanent_failure",
    },
    ErrorCategory.INVALID_INPUT: {
        "retry_with_backoff": False,
        "tell_model": True,
        "skip_tool_for_rest_of_run": False,
        "log_status": "invalid_input",
    },
    ErrorCategory.TOOL_NOT_AVAILABLE: {
        "retry_with_backoff": False,
        "tell_model": True,
        "skip_tool_for_rest_of_run": False,
        "log_status": "tool_not_available",
    },
}


# ── ToolError - the structured error a tool returns ────────────────────────
@dataclass
class ToolError:
    """A structured tool error. Wraps the message + category + recovery hint.

    Tools return either a plain string (success) or a ToolError (failure).
    The agent loop formats ToolError into a string the model can read, AND
    routes the failure to the right action_log status.
    """
    category: ErrorCategory
    message: str
    hint_to_model: Optional[str] = None

    def to_observation_string(self) -> str:
        """Format for the model to read on the next turn.

        Always prefixed with 'ERROR:' so the model sees it as a failure,
        and includes the category so the model can choose its next action.
        """
        parts = [f"ERROR[{self.category.value}]: {self.message}"]
        if self.hint_to_model:
            parts.append(f"Hint: {self.hint_to_model}")
        return " ".join(parts)


# ── Convenience constructors for each category ─────────────────────────────
def transient(message: str, hint: Optional[str] = None) -> ToolError:
    return ToolError(ErrorCategory.TRANSIENT, message, hint)


def permanent(message: str, hint: Optional[str] = None) -> ToolError:
    return ToolError(ErrorCategory.PERMANENT, message, hint)


def invalid_input(message: str, hint: Optional[str] = None) -> ToolError:
    return ToolError(ErrorCategory.INVALID_INPUT, message, hint)


def tool_not_available(tool_name: str, available: list) -> ToolError:
    return ToolError(
        ErrorCategory.TOOL_NOT_AVAILABLE,
        f"No tool named '{tool_name}'",
        hint=f"Available tools: {sorted(available)}",
    )


# ── classify(string) - parse an error string back into a category ──────────
# Useful when a legacy tool returns the old plain "ERROR: ..." format and
# the agent loop needs to categorise it after the fact.
def classify(error_string: str) -> ErrorCategory:
    """Best-effort classification of a legacy ERROR: string."""
    s = error_string.lower()
    if any(k in s for k in ("timeout", "5xx", "rate limit", "rate_limit", "throttl")):
        return ErrorCategory.TRANSIENT
    if any(k in s for k in ("invalid", "malformed", "disallowed", "out of range", "division by zero")):
        return ErrorCategory.INVALID_INPUT
    if any(k in s for k in ("no tool named", "tool not found", "unknown tool")):
        return ErrorCategory.TOOL_NOT_AVAILABLE
    return ErrorCategory.PERMANENT


if __name__ == "__main__":
    # Demonstrate each category
    print("=== Error Taxonomy ===")
    for cat in ErrorCategory:
        rec = RECOVERY[cat]
        print(f"\n{cat.value.upper()}")
        print(f"  retry_with_backoff:        {rec['retry_with_backoff']}")
        print(f"  tell_model:                {rec['tell_model']}")
        print(f"  skip_tool_for_rest_of_run: {rec['skip_tool_for_rest_of_run']}")
        print(f"  log_status:                {rec['log_status']}")

    print("\n=== Example error strings ===")
    e1 = transient("Network timeout to upstream", hint="The agent will retry automatically.")
    e2 = permanent("Wikipedia API has been deprecated", hint="Use web_search instead.")
    e3 = invalid_input("Division by zero", hint="Check the denominator before calling.")
    e4 = tool_not_available("calculate_taxes", available=["calculator", "web_search"])
    for e in [e1, e2, e3, e4]:
        print(f"  {e.to_observation_string()}")
