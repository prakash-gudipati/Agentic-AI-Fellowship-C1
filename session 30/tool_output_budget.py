"""
Session 30 - tool_output_budget.py

The context-window blow-up problem.

Every tool result becomes part of the conversation history. The next LLM
turn reads ALL prior tool results plus the new one. If web_search returns
50 KB of HTML, three turns later your context window is gone and the model
starts forgetting the original question.

This module provides budget-aware truncation - one helper that takes a
tool's raw output and a token budget, and returns a string that fits.

The truncation strategy is intentionally simple and observable:
  1. If the output fits, return it unchanged.
  2. If the output overflows, KEEP THE HEAD AND THE TAIL, replace the
     middle with a "[... truncated N chars ...]" marker.
     Head bias = first paragraph usually carries the answer.
     Tail bias = source attribution / final number usually at the end.
  3. Log the truncation event so the SRE can see when the agent's tool
     output is getting clipped.

Token budget arithmetic uses the 4-chars-per-token rule of thumb.
That's not exact, but it's stable and zero-dependency. For production,
swap in tiktoken or the provider's count_tokens API.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional


CHARS_PER_TOKEN = 4
DEFAULT_TOOL_OUTPUT_TOKEN_BUDGET = 500     # ~2000 chars per tool result
TRUNCATION_HEAD_FRACTION = 0.7             # keep 70% from start, 30% from end


@dataclass
class TruncationResult:
    """What happened during one truncation call."""
    text: str
    original_chars: int
    final_chars: int
    truncated: bool


def truncate_for_llm(
    raw_output: str,
    *,
    token_budget: int = DEFAULT_TOOL_OUTPUT_TOKEN_BUDGET,
    chars_per_token: int = CHARS_PER_TOKEN,
) -> TruncationResult:
    """Return a string that fits within the token budget.

    Strategy: head + tail with a truncation marker in the middle.
    Never silently truncates - the returned text always includes the marker
    when truncation happened, so the model knows what it's seeing is partial.
    """
    char_budget = token_budget * chars_per_token
    original = len(raw_output)

    if original <= char_budget:
        return TruncationResult(
            text=raw_output,
            original_chars=original,
            final_chars=original,
            truncated=False,
        )

    # Reserve ~40 chars for the marker
    marker_reserve = 50
    available = char_budget - marker_reserve
    head_chars = int(available * TRUNCATION_HEAD_FRACTION)
    tail_chars = available - head_chars

    head = raw_output[:head_chars]
    tail = raw_output[-tail_chars:]
    omitted = original - head_chars - tail_chars
    marker = f"\n[... truncated {omitted} chars ...]\n"

    truncated_text = head + marker + tail
    return TruncationResult(
        text=truncated_text,
        original_chars=original,
        final_chars=len(truncated_text),
        truncated=True,
    )


def wrap_tool_for_budget(
    tool_run_fn,
    *,
    token_budget: int = DEFAULT_TOOL_OUTPUT_TOKEN_BUDGET,
    log_truncation: bool = True,
):
    """Wrap a tool's run() function so its output is auto-truncated.

    Use it like:
        budgeted_run = wrap_tool_for_budget(tool.run, token_budget=300)
        tool.run = budgeted_run

    The original run is preserved inside the wrapper.
    """
    def wrapped(args):
        raw = tool_run_fn(args)
        if not isinstance(raw, str):
            raw = str(raw)
        result = truncate_for_llm(raw, token_budget=token_budget)
        if result.truncated and log_truncation:
            sys.stderr.write(
                f"[budget] truncated tool output: "
                f"{result.original_chars} -> {result.final_chars} chars\n"
            )
        return result.text
    return wrapped


# ── Recommended budgets per tool category (call sites can override) ────────
BUDGET_PRESETS = {
    "calculator":       50,       # always short - ~200 chars
    "web_search":       400,      # one paragraph
    "wikipedia":        800,      # 1-2 paragraphs
    "document_fetch":   1500,     # bigger windows for document tools
    "database_query":   600,      # typical query result summary
}


if __name__ == "__main__":
    # Demo: a fake "big" tool output, truncated to a small budget
    big = "FIRST_PARAGRAPH " + ("MIDDLE " * 500) + "LAST_PARAGRAPH_with_attribution"
    print(f"Original: {len(big)} chars")
    r = truncate_for_llm(big, token_budget=80)
    print(f"After truncation (budget=80 tokens = 320 chars): {r.final_chars} chars")
    print(f"Truncated: {r.truncated}")
    print("\n--- preview ---")
    print(r.text)
    print("--- end preview ---")
