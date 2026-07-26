"""
Session 32 — context_builder.py

THE DISCIPLINE — Context Engineering.

Memory is the MECHANISM. Context engineering is the DISCIPLINE of deciding,
turn by turn, what goes into the model's prompt and what stays out.

The builder reads every memory layer the agent owns and assembles the
single prompt the model will see this turn. It is the only place in the
codebase that knows the LAYOUT of the agent's context window:

    1. SYSTEM         — pinned instructions
    2. SUMMARY        — the rolling summary
    3. SEMANTIC HITS  — relevant past turns surfaced by vector recall
    4. WORKING        — the last K verbatim turns
    5. USER           — the latest user message

Why this layout?
  - System first because Anthropic / OpenAI both treat the system field
    as standing instructions with high adherence.
  - Summary before semantic hits so the model has a global anchor BEFORE
    it sees individual older turns.
  - Working memory last so the most recent context is closest to the
    user message — recency bias in attention helps here.
  - User message at the end so the model is reading the question last.

Token discipline:
  - Every section gets a maximum size in tokens. The builder reports the
    final breakdown so the instructor can show students the cost ledger.
  - If a section would overflow its budget, the builder TRUNCATES rather
    than silently drops the section. Truncation is loud; silent drops
    are how production prompts go wrong without an alert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from memory_types import SemanticHit, SummaryNote, Turn, estimate_tokens


# ----------------------------------------------------------------------------
# Output type
# ----------------------------------------------------------------------------


@dataclass
class BuiltContext:
    """
    The assembled prompt + a ledger of what went in.

    system   : the system message string
    messages : the Anthropic-shaped message list (role/content dicts)
    breakdown: per-section token counts; printed by trace_logger
    """

    system: str
    messages: List[dict]
    breakdown: dict = field(default_factory=dict)

    def total_tokens(self) -> int:
        return int(sum(self.breakdown.values()))


# ----------------------------------------------------------------------------
# ContextBuilder
# ----------------------------------------------------------------------------


@dataclass
class TokenBudgets:
    """Per-section soft caps. Tune these to your model's context window."""

    system: int = 200
    summary: int = 200
    semantic_hits: int = 400
    working: int = 800


class ContextBuilder:
    """
    Assembles the prompt from the four memory layers.

    The builder is a pure function over (system, summary, hits, working).
    No I/O, no LLM calls. This is what makes context engineering a
    DISCIPLINE — every decision is visible, deterministic, and auditable.
    """

    def __init__(self, budgets: Optional[TokenBudgets] = None) -> None:
        self.budgets = budgets or TokenBudgets()

    def build(
        self,
        system_text: str,
        summary: Optional[SummaryNote],
        semantic_hits: List[SemanticHit],
        working_turns: List[Turn],
        user_message: str,
    ) -> BuiltContext:
        breakdown = {}

        # ----- 1) SYSTEM -----
        system_block = self._truncate(system_text, self.budgets.system)
        breakdown["system"] = estimate_tokens(system_block)

        # ----- 2) SUMMARY -----
        summary_block = ""
        if summary and summary.text:
            summary_block = self._truncate(
                f"SUMMARY OF EARLIER CONVERSATION:\n{summary.text}",
                self.budgets.summary,
            )
        breakdown["summary"] = estimate_tokens(summary_block)

        # ----- 3) SEMANTIC HITS -----
        hits_block = self._render_hits(semantic_hits, self.budgets.semantic_hits)
        breakdown["semantic_hits"] = estimate_tokens(hits_block)

        # ----- 4) WORKING MEMORY -----
        # Working memory enters as a sequence of role-tagged messages so
        # the model can treat past assistant turns as model-authored.
        working_messages = self._render_working(working_turns, self.budgets.working)
        breakdown["working"] = sum(
            estimate_tokens(m["content"]) for m in working_messages
        )

        # ----- 5) USER (the new question) -----
        user_block = user_message.strip()
        breakdown["user"] = estimate_tokens(user_block)

        # Compose the message list.
        # The non-working preamble (summary + semantic) rides inside a
        # synthetic "user" message at the start, because Anthropic's
        # messages array does not have a dedicated "context" role.
        messages: List[dict] = []
        preamble_parts: List[str] = []
        if summary_block:
            preamble_parts.append(summary_block)
        if hits_block:
            preamble_parts.append(hits_block)
        if preamble_parts:
            messages.append({
                "role": "user",
                "content": "\n\n".join(preamble_parts),
            })
            # Pair the preamble with a one-word assistant acknowledgement
            # so the conversation alternates roles cleanly.
            messages.append({
                "role": "assistant",
                "content": "Understood.",
            })

        messages.extend(working_messages)
        messages.append({"role": "user", "content": user_block})

        return BuiltContext(
            system=system_block,
            messages=messages,
            breakdown=breakdown,
        )

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _render_hits(
        self, hits: List[SemanticHit], budget: int
    ) -> str:
        if not hits:
            return ""
        lines: List[str] = ["RELEVANT PRIOR TURNS (recalled from memory):"]
        for hit in hits:
            lines.append(
                f"- (score={hit.score:.2f}) [{hit.turn.role}] "
                f"{hit.turn.content}"
            )
        joined = "\n".join(lines)
        return self._truncate(joined, budget)

    def _render_working(
        self, turns: List[Turn], budget: int
    ) -> List[dict]:
        """
        Convert turns to Anthropic-shaped messages.

        We walk in order. If adding a turn would breach the budget we
        STOP rather than half-include it, because the model's view of
        the conversation must remain coherent.
        """

        out: List[dict] = []
        used = 0
        for t in turns:
            role = "assistant" if t.role == "assistant" else "user"
            cost = t.token_estimate or estimate_tokens(t.content)
            if used + cost > budget and out:
                break
            out.append({"role": role, "content": t.content})
            used += cost
        return out

    def _truncate(self, text: str, budget: int) -> str:
        if not text:
            return ""
        approx_chars = budget * 4  # mirror the 4-char/token estimator
        if len(text) <= approx_chars:
            return text
        truncated = text[: approx_chars - 25]
        return truncated.rstrip() + "  …[truncated]"
