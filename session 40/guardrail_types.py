"""Session 40 — shared types for the production safety layer.

Demonstrates the vocabulary every guardrail speaks: a Decision (what to do),
a Severity (how bad), and two report dataclasses that carry results up the stack.
WHY a shared types module: guardrails, the chain, and the agent all need to agree
on the SAME contract. One place to change it = one place to reason about safety.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional


class Decision(enum.Enum):
    """What a guardrail tells the chain to do with a request."""
    ALLOW = "ALLOW"      # let it through unchanged
    BLOCK = "BLOCK"      # stop here — refuse the request
    REDACT = "REDACT"    # rewrite the text, then continue


class Severity(enum.Enum):
    """How serious a tripped guardrail is — drives logging + alerting in prod."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class GuardrailResult:
    """The verdict from a SINGLE guardrail check."""
    name: str
    decision: Decision
    severity: Severity
    reason: str
    transformed_text: Optional[str] = None  # set only when decision == REDACT


@dataclass
class GuardrailReport:
    """The combined verdict from running a whole CHAIN of guardrails."""
    stage: str                                  # "input" or "output"
    results: List[GuardrailResult] = field(default_factory=list)
    final_decision: Decision = Decision.ALLOW
    final_text: str = ""

    @property
    def blocked(self) -> bool:
        """True when the chain decided to stop the request."""
        return self.final_decision == Decision.BLOCK


# --- tiny cost helpers (used by the trace + cost framing in the script) -------

def estimate_tokens(text: str) -> int:
    """Rough token count — ~4 chars per token is good enough for cost framing."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# Haiku-class blended price; only used for the "what does a guardrail cost" line.
_USD_PER_1K_TOKENS = 0.001


def estimate_cost_usd(text: str) -> float:
    """Estimate USD for processing `text` once — for the cost-of-safety framing."""
    return (estimate_tokens(text) / 1000.0) * _USD_PER_1K_TOKENS
