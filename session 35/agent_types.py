"""
Session 35 — agent_types.py

Typed dataclasses for the hierarchical + debate multi-agent system.

KEY IDEAS this file encodes:
  - A Message is the unit of inter-agent communication. Every cross-agent
    exchange in S35 flows through a Message — never free-form chat.
  - A Verdict is the unit of a Judge's choice in Best-of-N.
  - A ConsensusReport is the unit of a Moderator's synthesis in Debate.
  - Cost helpers reused from S34 — multi-agent multiplies LLM calls;
    we keep counting honest.

Phase 5 (v2.4) build convention: every dataclass is small, frozen where
possible, and never imports from agents/*. The dependency arrow points
INTO agent_types.py, never out of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ===========================================================================
# Roles
# ===========================================================================


class Role:
    """Role name constants — used as the opener-phrase root in every
    system prompt. See prompts.py for the actual prompts."""

    DIRECTOR = "director"
    RESEARCH_MANAGER = "research_manager"
    EDITORIAL_MANAGER = "editorial_manager"
    RESEARCHER = "researcher"
    WRITER = "writer"
    FACT_CHECKER = "fact_checker"

    # Debate panel
    BULL = "bull_panelist"
    BEAR = "bear_panelist"
    NEUTRAL = "neutral_panelist"
    MODERATOR = "moderator"

    # Competitive panel
    CANDIDATE = "candidate_writer"
    JUDGE = "judge"


# ===========================================================================
# Inter-agent messages — PROD PATTERN: Inter-Agent Message Schema
# ===========================================================================


@dataclass
class Message:
    """Typed inter-agent message. Replaces the free-form 'append to a
    shared list' anti-pattern.

    Fields are required by name — the wire format is JSON-serialisable.
    """

    sender: str           # Role name (see Role) or "system" / "user"
    recipient: str        # Role name of the intended receiver
    intent: str           # One of: ASSIGN, REPORT, REVIEW, ESCALATE, ACK
    subject: str          # One-line summary — what this message is about
    payload: Dict[str, Any] = field(default_factory=dict)
    round_num: int = 0    # Which coordination round produced this
    msg_id: str = ""

    # Allowed intents — enforce at construction time.
    ALLOWED_INTENTS = {"ASSIGN", "REPORT", "REVIEW", "ESCALATE", "ACK"}

    def __post_init__(self) -> None:
        if self.intent not in self.ALLOWED_INTENTS:
            raise ValueError(
                f"Message.intent must be one of {self.ALLOWED_INTENTS}, "
                f"got '{self.intent}'."
            )
        if not self.sender or not self.recipient:
            raise ValueError("Message requires non-empty sender + recipient.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "intent": self.intent,
            "subject": self.subject,
            "payload": self.payload,
            "round_num": self.round_num,
            "msg_id": self.msg_id,
        }


# ===========================================================================
# Domain dataclasses — facts, drafts, verdicts
# ===========================================================================


@dataclass
class Fact:
    """A single factual claim with a source name. Same shape as S34."""

    text: str
    source: str


@dataclass
class Argument:
    """One panelist's contribution to a debate round."""

    panelist: str         # Role name: BULL / BEAR / NEUTRAL
    claim: str            # The position being argued
    evidence: List[str] = field(default_factory=list)
    round_num: int = 0


@dataclass
class ConsensusReport:
    """The Moderator's structured synthesis output."""

    agreed_points: List[str]
    disagreements: List[str]
    final_position: str   # One paragraph — the consensus
    confidence: float     # 0.0 - 1.0


@dataclass
class CandidateDraft:
    """A single Candidate Writer's output in the competitive demo."""

    candidate_id: str
    draft: str
    style: str            # One-word style tag, e.g. "concise", "detailed"


@dataclass
class JudgeVerdict:
    """The Judge's output in the competitive demo."""

    winner_id: str
    rationale: str
    scores: Dict[str, Dict[str, float]]   # candidate_id -> {criterion: score}


# ===========================================================================
# Cost accounting (reused shape from S34)
# ===========================================================================


# Anthropic Haiku 4.5 default rates as of session build (USD per 1M tokens).
# These are illustrative — the math in the slides uses these constants.
COST_INPUT_PER_M = 1.00
COST_OUTPUT_PER_M = 5.00
USD_TO_INR = 83.0


def estimate_call_cost_usd(input_tokens: int, output_tokens: int) -> float:
    in_cost = (input_tokens / 1_000_000.0) * COST_INPUT_PER_M
    out_cost = (output_tokens / 1_000_000.0) * COST_OUTPUT_PER_M
    return in_cost + out_cost


def usd_to_inr(usd: float) -> float:
    return usd * USD_TO_INR


# ===========================================================================
# Result containers
# ===========================================================================


@dataclass
class HierarchicalResult:
    """What the Director returns at the end of a hierarchical run."""

    final_answer: str
    messages: List[Message]
    total_llm_calls: int
    terminated_reason: str        # "all_done", "max_rounds", "escalated"


@dataclass
class DebateResult:
    """What the Moderator returns at the end of a debate."""

    consensus: ConsensusReport
    transcript: List[Argument]
    total_llm_calls: int
    rounds_used: int


@dataclass
class CompetitiveResult:
    """What the Judge returns at the end of a Best-of-N run."""

    verdict: JudgeVerdict
    candidates: List[CandidateDraft]
    total_llm_calls: int
