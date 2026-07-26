"""
Session 34 — agent_types.py

Typed records shared across the multi-agent crew.

These are the records every agent reads and writes. Keeping them in
one file makes the data flow legible — the orchestrator emits Tasks,
workers emit Verdicts/Facts/Drafts, the Scratchpad stores everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Task:
    """A unit of work the orchestrator hands to a worker."""

    task_id: str
    worker: str   # e.g. "researcher", "writer", "fact_checker"
    instruction: str
    inputs: Dict[str, str] = field(default_factory=dict)
    done: bool = False
    result_key: str = ""   # where in the Scratchpad the result lands


@dataclass
class Fact:
    """One fact pulled by the researcher."""

    text: str
    source: str = ""


@dataclass
class Critique:
    """The fact-checker's verdict on a draft."""

    verdict: str           # "ACCEPT" or "REVISE"
    issues: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def accepted(self) -> bool:
        return self.verdict.upper() == "ACCEPT"


@dataclass
class CrewResult:
    """End-of-turn record. Everything the walkthrough needs in one object."""

    user_request: str
    final_text: str
    rounds: int = 0
    max_rounds: int = 5
    terminated_reason: str = ""   # "all_done" / "quality_met" / "max_rounds"
    facts: List[Fact] = field(default_factory=list)
    critique: Optional[Critique] = None
    elapsed_seconds: float = 0.0
    llm_calls: int = 0


# ---------------------------------------------------------------------------
# Token cost — same pricing as S33 for consistency with the cost slide.
# ---------------------------------------------------------------------------


HAIKU_INPUT_PER_M = 1.00
HAIKU_OUTPUT_PER_M = 5.00
INR_PER_USD = 83.0


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * HAIKU_INPUT_PER_M + (
        output_tokens / 1_000_000
    ) * HAIKU_OUTPUT_PER_M


def estimate_cost_inr(input_tokens: int, output_tokens: int) -> float:
    return estimate_cost_usd(input_tokens, output_tokens) * INR_PER_USD
