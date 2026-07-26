"""
Session 32 — eviction.py

PROD PATTERN — Memory Eviction Policy.

Three concrete eviction policies behind a single Strategy-pattern interface.
WorkingMemory holds a reference to ONE policy at construction time and
defers every "who leaves?" decision to it.

Why a named pattern instead of an `if` block? Because the policy choice is
the loudest production decision you make about an agent's memory:

  - RecencyPolicy        — newest in, oldest out. The default. Simple,
                           predictable, equivalent to LangChain's
                           ConversationBufferWindowMemory.
  - ImportancePolicy     — score every turn, evict the lowest. Used when
                           you tag certain turns as "must keep" (system
                           instructions, user identity, in-progress task).
  - HybridPolicy         — keep the last K turns AND any turn flagged
                           important; evict the rest by recency.

Every policy returns the INDEX of the victim turn so the buffer can pop
it without scanning. None means "refuse to evict" — used by the min_keep
floor to prevent buffer collapse.
"""

from __future__ import annotations

from typing import List, Optional, Protocol

from memory_types import Turn


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------


class EvictionPolicy(Protocol):
    def name(self) -> str: ...
    def pick_victim(self, turns: List[Turn]) -> Optional[int]: ...


# ----------------------------------------------------------------------------
# RecencyPolicy
# ----------------------------------------------------------------------------


class RecencyPolicy:
    """
    Evict the oldest turn first.

    min_keep is a floor — eviction stops once the buffer is down to
    min_keep turns regardless of token budget. Without this floor the
    buffer can collapse to zero turns and the agent forgets the goal.
    """

    def __init__(self, min_keep: int = 2) -> None:
        self.min_keep = min_keep

    def name(self) -> str:
        return "recency"

    def pick_victim(self, turns: List[Turn]) -> Optional[int]:
        if len(turns) <= self.min_keep:
            return None
        return 0


# ----------------------------------------------------------------------------
# ImportancePolicy
# ----------------------------------------------------------------------------


class ImportancePolicy:
    """
    Evict the turn with the lowest importance score.

    Importance lives in turn.metadata["importance"], an int in [0..10].
    Default importance is 5 — used when the agent did not score the turn.

    Ties broken by recency (older loses).

    Pin protection: turns with metadata["pinned"]=True are never evicted.
    A SystemPrompt-style instruction turn typically gets pinned=True.
    """

    def __init__(self, min_keep: int = 2) -> None:
        self.min_keep = min_keep

    def name(self) -> str:
        return "importance"

    def pick_victim(self, turns: List[Turn]) -> Optional[int]:
        if len(turns) <= self.min_keep:
            return None

        worst_index: Optional[int] = None
        worst_score = 11  # any real score (0-10) beats this

        for i, turn in enumerate(turns):
            if turn.metadata.get("pinned"):
                continue
            score = int(turn.metadata.get("importance", 5))
            # Strict < so ties go to the older turn (lower index).
            if score < worst_score:
                worst_score = score
                worst_index = i

        return worst_index


# ----------------------------------------------------------------------------
# HybridPolicy
# ----------------------------------------------------------------------------


class HybridPolicy:
    """
    Keep the last K turns and any pinned turn. Evict the rest by recency.

    The most common production policy. It says: "always show the model the
    last K turns verbatim, plus any turn I have explicitly told you to
    pin. Everything older is fair game."

    k is the recency window. min_keep applies as the absolute floor.
    """

    def __init__(self, k: int = 4, min_keep: int = 2) -> None:
        self.k = k
        self.min_keep = min_keep

    def name(self) -> str:
        return f"hybrid(k={self.k})"

    def pick_victim(self, turns: List[Turn]) -> Optional[int]:
        if len(turns) <= self.min_keep:
            return None

        # Indices of the last K turns — these are protected by recency.
        protected = set(range(max(0, len(turns) - self.k), len(turns)))

        for i, turn in enumerate(turns):
            if i in protected:
                continue
            if turn.metadata.get("pinned"):
                continue
            return i  # first eligible turn = oldest unprotected

        return None  # everything left is either pinned or in the K window
