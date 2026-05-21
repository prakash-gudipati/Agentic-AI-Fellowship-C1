"""
Session 29 — parser.py

Pull structured signals out of the model's free-form ReAct output.

Why does this exist as its own module?
  - The agent loop reads the model's reply and asks two questions:
      1. Did it emit an Action I should execute?
      2. Did it emit a Final Answer I should return to the user?
    Both questions are string-parsing problems. Keeping them out of the
    loop file keeps the loop file focused on orchestration.
  - If we ever change the ReAct format (different action syntax, JSON
    instead of brackets, a different stop token), we change ONE file.
    That's the single-responsibility pattern from S3 applied at module
    granularity.

Format recap (matches prompts.py):

  Thought: <reasoning>
  Action: <tool_name>[<input>]
  ...
  Final Answer: <answer>
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ── Regex contracts ─────────────────────────────────────────────────────────
# The Action line: a tool name (letters/digits/underscores) followed by
# square-bracketed input. We use a non-greedy match on the input so a
# closing bracket inside the input doesn't trip us up — but the LAST
# matching ] still wins, which is what we want.
ACTION_PATTERN = re.compile(
    r"^\s*Action:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[(.*)\]\s*$",
    re.MULTILINE,
)

# The Final Answer line: anything after the colon, to end of string.
FINAL_ANSWER_PATTERN = re.compile(
    r"Final\s+Answer\s*:\s*(.+?)\s*$",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParsedAction:
    """A tool invocation the agent intends to make."""
    tool_name: str
    tool_input: str


def parse_action(model_text: str) -> Optional[ParsedAction]:
    """Return the LAST Action line in the text, or None if there isn't one.

    Why the LAST one: if the LLM emits two Action lines in a single reply
    (it shouldn't, but it sometimes does), the last one reflects its latest
    decision. We honour that and ignore earlier drafts.
    """
    matches = list(ACTION_PATTERN.finditer(model_text))
    if not matches:
        return None

    last = matches[-1]
    return ParsedAction(
        tool_name=last.group(1).strip(),
        tool_input=last.group(2).strip(),
    )


def parse_final_answer(model_text: str) -> Optional[str]:
    """Return the text following 'Final Answer:' if present, else None."""
    match = FINAL_ANSWER_PATTERN.search(model_text)
    if not match:
        return None
    return match.group(1).strip()
