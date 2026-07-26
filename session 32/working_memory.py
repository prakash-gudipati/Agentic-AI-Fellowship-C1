"""
Session 32 — working_memory.py

The working-memory buffer.

This is the SHORT-TERM, VERBATIM layer of the agent's memory system. Every
turn the user types or the model produces lands here first. The buffer
holds the most recent N tokens of conversation in full — no compression,
no embedding, no scoring. That's the whole point of a working-memory
layer: instant recall of the recent past, no lookup cost.

What lives here is bounded by a TOKEN BUDGET. Anthropic's API charges per
input token on every turn, so an unbounded conversation grows linearly in
cost. The token budget is the wall that stops that growth — when a new
turn would push the buffer past the budget, the eviction policy fires.

The summariser (summariser.py) wraps the eviction trigger: turns that are
about to be evicted are first folded into the rolling summary so their
content is preserved at a fraction of the cost.

Public surface:
  - WorkingMemory.append(turn)             -> EvictionResult
  - WorkingMemory.turns()                  -> list[Turn]
  - WorkingMemory.token_count()            -> int
  - WorkingMemory.snapshot()               -> list[Turn] (defensive copy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from eviction import EvictionPolicy, RecencyPolicy
from memory_types import EvictionEvent, Turn, estimate_tokens


# ----------------------------------------------------------------------------
# Result type returned by every append
# ----------------------------------------------------------------------------


@dataclass
class EvictionResult:
    """
    What an append() call did.

    evicted is the list of Turn objects that were removed from the buffer.
    The caller (usually the Summariser) picks these up, summarises them,
    and the summary is stored in the SummaryNote list on the agent.

    events mirrors the same eviction action as an EvictionEvent record for
    the SessionStateSnapshot trace.
    """

    evicted: List[Turn] = field(default_factory=list)
    events: List[EvictionEvent] = field(default_factory=list)


# ----------------------------------------------------------------------------
# WorkingMemory
# ----------------------------------------------------------------------------


class WorkingMemory:
    """
    Token-bounded buffer of the most recent turns.

    Constructor parameters
      token_budget   — soft cap. When append() would push total tokens
                       past this, eviction runs.
      max_turns      — hard cap on number of turns. Belt-and-braces; some
                       cohorts care about turn counts more than tokens.
      policy         — EvictionPolicy implementation. Defaults to
                       RecencyPolicy(min_keep=2), which mirrors the
                       LangChain ConversationBufferWindowMemory behaviour
                       students will recognise in Session 37.

    The "min_keep" floor is important. Without it, a single very long
    turn could blow the budget and evict everything including itself,
    leaving the agent with no context at all. Students hit this exact
    bug in the exercise — name the bug "buffer collapse" in the trace.
    """

    def __init__(
        self,
        token_budget: int = 800,
        max_turns: int = 30,
        policy: Optional[EvictionPolicy] = None,
        on_evict: Optional[Callable[[List[Turn]], None]] = None,
    ) -> None:
        self.token_budget = token_budget
        self.max_turns = max_turns
        self.policy = policy or RecencyPolicy(min_keep=2)
        self._on_evict = on_evict
        self._turns: List[Turn] = []

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def append(self, turn: Turn) -> EvictionResult:
        """
        Add a turn to the buffer and run eviction if needed.

        We compute the token estimate exactly once per turn — at append
        time. Re-estimating on every read would be wasteful, and the
        content of a turn never changes once it lands.
        """

        if turn.token_estimate == 0:
            turn.token_estimate = estimate_tokens(turn.content)

        self._turns.append(turn)
        return self._enforce_budget()

    def clear(self) -> None:
        self._turns = []

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def turns(self) -> List[Turn]:
        """Direct (live) view of the turns currently in the buffer."""

        return self._turns

    def snapshot(self) -> List[Turn]:
        """Defensive copy — safe to keep around past the next append."""

        return list(self._turns)

    def token_count(self) -> int:
        return sum(t.token_estimate for t in self._turns)

    def turn_count(self) -> int:
        return len(self._turns)

    # ------------------------------------------------------------------
    # Eviction core
    # ------------------------------------------------------------------

    def _enforce_budget(self) -> EvictionResult:
        """
        Evict until both budgets are satisfied.

        Order of checks matters:
          1) token budget first — usually the binding constraint
          2) max_turns second — a safety net for pathological
             cases where every turn is one short word

        We loop because evicting one turn might still leave us over budget
        if the buffer is full of long turns.
        """

        evicted_turns: List[Turn] = []
        events: List[EvictionEvent] = []

        while (
            self.token_count() > self.token_budget
            or self.turn_count() > self.max_turns
        ):
            victim_index = self.policy.pick_victim(self._turns)
            if victim_index is None:
                # Policy refuses to evict (min_keep floor). Stop —
                # the buffer is as small as the policy allows.
                break

            victim = self._turns.pop(victim_index)
            evicted_turns.append(victim)
            reason = (
                "token_budget"
                if self.token_count() + victim.token_estimate > self.token_budget
                else "max_turns"
            )
            events.append(
                EvictionEvent(
                    turn_id=victim.turn_id,
                    reason=reason,
                    policy=self.policy.name(),
                )
            )

        if evicted_turns and self._on_evict is not None:
            self._on_evict(evicted_turns)

        return EvictionResult(evicted=evicted_turns, events=events)
