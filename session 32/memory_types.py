"""
Session 32 — memory_types.py

The data types that move through the memory system.

Every memory layer in this build (working / episodic / semantic / long-term)
serialises down to a list of Turn objects. The four production patterns then
operate on those turns:

  - WorkingMemory      keeps the most recent turns verbatim
  - Summariser         compresses old turns into a rolling SummaryNote
  - EvictionPolicy     decides which turns leave working memory and when
  - SemanticMemory     embeds turns so old conversations are searchable
  - SessionState       packages everything into one JSON artifact

The dataclasses here are deliberately small. They do not import anything
outside the standard library so trace_logger / agent.py can import them
without pulling in the Anthropic SDK or numpy.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------------
# Turn — the atomic unit of conversation
# ----------------------------------------------------------------------------


@dataclass
class Turn:
    """
    One exchange between the user and the agent (or the agent and a tool).

    role values:
      "user"         — text typed by the user
      "assistant"    — text produced by the model
      "tool_call"    — model asked to call a tool (recorded for the trace)
      "tool_result"  — observation that came back from the tool

    The token_estimate field is filled in lazily by WorkingMemory the first
    time a turn enters the buffer. It is intentionally an estimate — the
    point of the token budget is to make eviction predictable, not exact.
    """

    role: str
    content: str
    token_estimate: int = 0
    created_at: float = field(default_factory=time.time)
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        return cls(**data)


# ----------------------------------------------------------------------------
# SummaryNote — output of the rolling summariser
# ----------------------------------------------------------------------------


@dataclass
class SummaryNote:
    """
    A compressed view of N evicted turns produced by the Summariser.

    covers_turn_ids lets the trace prove WHICH raw turns this summary stands
    in for. In production you keep this so you can re-expand a summary back
    to its raw turns when you need to debug a specific run.
    """

    text: str
    covers_turn_ids: List[str] = field(default_factory=list)
    token_estimate: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SummaryNote":
        return cls(**data)


# ----------------------------------------------------------------------------
# SemanticHit — one result returned by the semantic memory layer
# ----------------------------------------------------------------------------


@dataclass
class SemanticHit:
    """
    A retrieved memory plus its similarity score.

    score is in [-1.0, 1.0] for cosine similarity (we use a centered
    embedding so negatives are possible). The context builder uses this
    score to decide whether the hit clears the inclusion threshold.
    """

    turn: Turn
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"turn": self.turn.to_dict(), "score": self.score}


# ----------------------------------------------------------------------------
# EvictionEvent — what fell out of working memory and why
# ----------------------------------------------------------------------------


@dataclass
class EvictionEvent:
    """
    Recorded every time the eviction policy drops a turn from working memory.

    Kept so the instructor can show students the eviction trace during the
    walkthrough — "look, here is the turn that left at 14:32 because the
    token budget was breached." Without this trace the eviction pattern is
    invisible to the student.
    """

    turn_id: str
    reason: str  # "token_budget" | "max_turns" | "importance_score" | "ttl"
    policy: str  # name of the policy that fired
    at: float = field(default_factory=time.time)


# ----------------------------------------------------------------------------
# SessionStateSnapshot — the full serialisable artifact
# ----------------------------------------------------------------------------


@dataclass
class SessionStateSnapshot:
    """
    Snapshot of every memory layer at one moment in time.

    PROD PATTERN — Session State as Artifact.

    This is the object the agent serialises to disk at the end of a run and
    re-hydrates at the start of the next one. It deliberately captures
    every layer so a replayed run does not silently drift from the original.
    """

    session_id: str
    created_at: float
    working_turns: List[Turn]
    summary_notes: List[SummaryNote]
    semantic_turn_ids: List[str]
    eviction_log: List[EvictionEvent]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "working_turns": [t.to_dict() for t in self.working_turns],
            "summary_notes": [s.to_dict() for s in self.summary_notes],
            "semantic_turn_ids": list(self.semantic_turn_ids),
            "eviction_log": [asdict(e) for e in self.eviction_log],
            "metadata": dict(self.metadata),
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "SessionStateSnapshot":
        data = json.loads(raw)
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            working_turns=[Turn.from_dict(t) for t in data["working_turns"]],
            summary_notes=[SummaryNote.from_dict(s) for s in data["summary_notes"]],
            semantic_turn_ids=list(data["semantic_turn_ids"]),
            eviction_log=[EvictionEvent(**e) for e in data["eviction_log"]],
            metadata=dict(data.get("metadata", {})),
        )


# ----------------------------------------------------------------------------
# Token estimator
# ----------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """
    Cheap, deterministic token estimator.

    Production code reaches for tiktoken / anthropic.token_count. For
    pedagogy we approximate at 4 chars/token. The point of the token
    budget is to make eviction predictable, not exact — an estimator
    that students can run offline is the right trade for a Phase 5
    walkthrough session.

    Empirical sanity: English text averages ~3.7 chars/token on the
    Anthropic and OpenAI tokenisers. 4 is close enough.
    """

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
