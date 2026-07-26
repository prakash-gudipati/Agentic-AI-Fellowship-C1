"""
Session 32 — summariser.py

PROD PATTERN — Conversation Summarisation (rolling summary buffer).

When the eviction policy fires, the evicted turns are not just dropped —
they are folded into a rolling summary that travels alongside the verbatim
working-memory buffer for the rest of the session.

The agent prompt then becomes:

    [system instructions]
    [SUMMARY of earlier conversation]   <-- the rolling summary
    [last K turns verbatim]              <-- the working memory
    [USER's latest message]

That single change — from a sliding window with hard drops to a sliding
window with summarised tail — is the difference between an agent that
"forgets it once existed" at turn 6 and an agent that can carry a 50-turn
session under a fixed token budget.

What's named here:
  - Summariser.fold(evicted, prior)  -> SummaryNote
      Produce a new rolling summary by merging the prior summary with
      the freshly evicted turns. Returns the new summary; caller stores it.

  - Summariser uses an LLMClient at runtime, but FAKE_LLM=1 will
    deterministically produce a canned summary so the demos are
    reproducible offline.

Why a rolling summary instead of one-shot summary at the end?
  - We need the summary to be valid AT EVERY TURN, not just at session
    end, because the agent reads it on every API call.
  - One-shot summarisation at session end is a different pattern called
    "episodic memory consolidation" — useful for long-term storage but
    not the right tool for the live context window.
"""

from __future__ import annotations

from typing import List, Optional

from llm_client import LLMClient
from memory_types import SummaryNote, Turn, estimate_tokens


# ----------------------------------------------------------------------------
# Prompt — small, deterministic, model-agnostic
# ----------------------------------------------------------------------------


SUMMARISER_SYSTEM_PROMPT = (
    "You are the SUMMARISER inside an agent's memory system. Your job is "
    "to maintain a rolling summary of the conversation so far.\n\n"
    "You will receive:\n"
    "  - PRIOR_SUMMARY: the summary as of the last update (may be empty).\n"
    "  - NEW_TURNS: turns that have just left the working-memory buffer.\n\n"
    "Produce a new summary that:\n"
    "  - Preserves user-stated facts, identities, preferences, and goals.\n"
    "  - Preserves any tool outputs that are still relevant.\n"
    "  - Drops chit-chat, repetition, and meta-commentary.\n"
    "  - Stays under 6 short sentences total.\n\n"
    "Respond with the summary text ONLY. No preamble, no quote marks, "
    "no bullets unless the source had them."
)


# ----------------------------------------------------------------------------
# Summariser
# ----------------------------------------------------------------------------


class Summariser:
    """
    Maintains the rolling summary.

    Stateless from the caller's perspective — fold() takes the prior
    summary in and returns a new one. The caller (agent.py) holds the
    canonical reference.

    target_tokens controls how aggressive the summary is. Lower = cheaper
    to read on every turn, higher = preserves more nuance.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        target_tokens: int = 120,
    ) -> None:
        self.client = client or LLMClient()
        self.target_tokens = target_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fold(
        self,
        evicted: List[Turn],
        prior: Optional[SummaryNote] = None,
    ) -> SummaryNote:
        """
        Produce the next rolling summary.

        If evicted is empty AND prior is non-empty, returns prior unchanged.
        If both are empty, returns an empty summary.
        """

        if not evicted and prior is not None:
            return prior

        prior_text = prior.text if prior else ""
        prior_ids = prior.covers_turn_ids if prior else []

        user_prompt = self._render_user_prompt(prior_text, evicted)
        new_text = self.client.complete(
            system=SUMMARISER_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=max(160, self.target_tokens + 60),
        ).strip()

        if not new_text:
            # If the model failed to produce a summary, keep the prior.
            # Drop the new turns silently rather than corrupt the state.
            return prior or SummaryNote(text="", covers_turn_ids=[])

        return SummaryNote(
            text=new_text,
            covers_turn_ids=list(prior_ids) + [t.turn_id for t in evicted],
            token_estimate=estimate_tokens(new_text),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _render_user_prompt(
        self, prior_text: str, evicted: List[Turn]
    ) -> str:
        lines: List[str] = []
        lines.append("PRIOR_SUMMARY:")
        lines.append(prior_text if prior_text else "(empty)")
        lines.append("")
        lines.append("NEW_TURNS (just left the working-memory buffer):")
        if evicted:
            for t in evicted:
                lines.append(f"- [{t.role}] {t.content}")
        else:
            lines.append("(none)")
        lines.append("")
        lines.append(
            f"Produce the updated rolling summary now, "
            f"target length ~{self.target_tokens} tokens."
        )
        return "\n".join(lines)
