"""
Session 30 — description_patterns.py

The five named patterns for writing a tool description.

Why does this exist as its own module?
  - tools.py shows three real working tools. This file shows FIVE PATTERNS
    you apply when writing any tool description. It's the design vocabulary
    you carry from this session into every agent you build for the rest of
    your career.
  - Keeping the patterns separate from the live tools means students can
    open this file alone, read each pattern once, and have a sharp mental
    model. Mixing patterns into tools.py would bury them under code.

The five patterns:

  1. THE JD PATTERN (job description)
     - Default. Use when the tool has one dominant intent.
     - Three sentences: WHAT it does · WHEN to use it · ONE example.
     - Every tool in tools.py uses this pattern as its base.

  2. EXAMPLE-DRIVEN
     - Use when the input format is unusual or easy to get wrong.
     - Description leans heavily on a concrete example INSIDE the description.
     - The calculator's description is example-driven on top of JD.

  3. CONTRAST
     - Use when TWO tools could plausibly answer the same question.
     - Description names what the OTHER tool does NOT do.
     - wikipedia_summary's "NOT for current statistics — use web_search for those"
       is a contrast clause.

  4. GUARDRAIL
     - Use when there is one obvious wrong way to invoke the tool.
     - Description includes a "ONLY for X, NOT for Y" guardrail clause.

  5. ANTI-CONFUSION
     - Use when the tool name could mean different things in different domains.
     - Description anchors the domain explicitly — "in real-estate context"
       or "for income tax (not corporate)".

Each pattern below shows the SAME tool described two ways:
the bare-bones JD version, then the version with the additional pattern applied.
"""

from __future__ import annotations

from typing import Dict


# ── PATTERN 1 — THE JD PATTERN (baseline) ───────────────────────────────────
JD_PATTERN = {
    "name": "The JD Pattern (job description)",
    "when_to_use": (
        "Default for every tool. Use when the tool has one dominant intent. "
        "Three sentences: what, when, one example."
    ),
    "example_tool_name": "web_search",
    "before_description": "Searches the web.",
    "after_description": (
        "Look up a single fact on the public web. "
        "Use this when the question depends on a current statistic or recent event. "
        "Example query: 'GDP of India 2024'."
    ),
}

# ── PATTERN 2 — EXAMPLE-DRIVEN ──────────────────────────────────────────────
EXAMPLE_DRIVEN_PATTERN = {
    "name": "Example-Driven",
    "when_to_use": (
        "Use when the input format is unusual or easy to get wrong. "
        "Lean on a concrete in-description example so the model imitates it."
    ),
    "example_tool_name": "calculator",
    "before_description": (
        "Evaluate a single arithmetic expression and return the numeric result."
    ),
    "after_description": (
        "Evaluate a single arithmetic expression and return the numeric result. "
        "Input MUST be a math expression using only digits and + - * / ( ). "
        "Example expression: '3937011000000 / 1441719852' (no commas, no units, no words)."
    ),
}

# ── PATTERN 3 — CONTRAST ────────────────────────────────────────────────────
CONTRAST_PATTERN = {
    "name": "Contrast",
    "when_to_use": (
        "Use when two tools in your catalog could plausibly answer the same question. "
        "Write each description against the OTHER — name what the other tool does NOT do."
    ),
    "example_tool_name": "wikipedia_summary",
    "before_description": (
        "Return a one-to-two-paragraph encyclopedia summary of a named topic."
    ),
    "after_description": (
        "Return a one-to-two-paragraph encyclopedia summary of a named topic. "
        "Use this when the question asks WHO or WHAT something is. "
        "NOT for current statistics or dated figures — use `web_search` for those. "
        "Example topic: 'Alan Turing'."
    ),
}

# ── PATTERN 4 — GUARDRAIL ───────────────────────────────────────────────────
GUARDRAIL_PATTERN = {
    "name": "Guardrail",
    "when_to_use": (
        "Use when there is one obvious wrong way to invoke the tool that the model "
        "WILL try unless you tell it not to."
    ),
    "example_tool_name": "tax_take_home",
    "before_description": "Compute after-tax income.",
    "after_description": (
        "Compute the after-tax take-home from an annual gross income, per country. "
        "Use this for INDIVIDUAL INCOME TAX only — NOT for corporate tax, NOT for capital gains, "
        "NOT for tax-policy lookups (use web_search for those). "
        "Example input: gross_income=1200000, country_code='IN'."
    ),
}

# ── PATTERN 5 — ANTI-CONFUSION ──────────────────────────────────────────────
ANTI_CONFUSION_PATTERN = {
    "name": "Anti-Confusion",
    "when_to_use": (
        "Use when the tool name could plausibly mean different things in different "
        "domains. Anchor the domain inside the description so the model doesn't drift."
    ),
    "example_tool_name": "search_listings",
    "before_description": "Searches listings.",
    "after_description": (
        "Search the active REAL-ESTATE property listings index by city and price band. "
        "This is NOT a generic web search and NOT a job-listing search — it queries "
        "the property database only. "
        "Example: city='Bangalore', price_max_inr=15000000."
    ),
}


# ── PATTERN 6 — SELF-DISABLING (S30 v3 addition) ────────────────────────────
SELF_DISABLING_PATTERN = {
    "name": "Self-Disabling",
    "when_to_use": (
        "Use when a tool can plausibly be called in an infinite loop "
        "(repeatedly searching with no progress, repeatedly retrying a "
        "broken upstream). Embed a self-disable clause IN the description "
        "so the model knows when to stop trying and answer with what it has."
    ),
    "example_tool_name": "web_search",
    "before_description": (
        "Look up a single fact on the public web. "
        "Use this when the question depends on a current statistic. "
        "Example query: 'GDP of India 2024'."
    ),
    "after_description": (
        "Look up a single fact on the public web. "
        "Use this when the question depends on a current statistic. "
        "If you have already called this tool twice on the same topic without "
        "getting a useful result, STOP and answer with what you have - do not "
        "call it a third time. "
        "Example query: 'GDP of India 2024'."
    ),
}


ALL_PATTERNS: list[Dict[str, str]] = [
    JD_PATTERN,
    EXAMPLE_DRIVEN_PATTERN,
    CONTRAST_PATTERN,
    GUARDRAIL_PATTERN,
    ANTI_CONFUSION_PATTERN,
    SELF_DISABLING_PATTERN,
]


def render_patterns_as_text() -> str:
    """Pretty-print the five patterns. Used by the demo to make the cheatsheet
    show up in the terminal alongside the live agent runs."""
    lines = []
    for i, pattern in enumerate(ALL_PATTERNS, start=1):
        lines.append(f"\n── PATTERN {i}: {pattern['name'].upper()} ──")
        lines.append(f"  When to use: {pattern['when_to_use']}")
        lines.append(f"  Tool example: {pattern['example_tool_name']}")
        lines.append(f"  BEFORE: {pattern['before_description']!r}")
        lines.append(f"  AFTER:  {pattern['after_description']!r}")
    return "\n".join(lines)


# ── ANTI-PATTERNS — what NOT to do (paired counter-examples) ────────────────
# These are the eight anti-patterns from the gallery slide. Keeping them in
# code form so students can see them as data, not just on a slide.

ANTI_PATTERNS: list[Dict[str, str]] = [
    {
        "label": "Single-letter parameter",
        "bad":  '{"properties": {"x": {"type": "string"}}}',
        "breaks": "Model passes the wrong thing into x — has no signal.",
        "fix":  "Name the parameter for the domain: 'expression', 'query', 'topic'.",
    },
    {
        "label": "Vague verb name",
        "bad":  'name="process", description="Processes data."',
        "breaks": "LLM picks it for anything. Description gives zero signal.",
        "fix":  "Verb the specific output: 'compute_take_home', 'summarise_listing'.",
    },
    {
        "label": "Missing required",
        "bad":  '{"properties": {"query": {...}}}  # no required key',
        "breaks": "Model sometimes calls with no args — API doesn't enforce.",
        "fix":  "Always list `required` explicitly.",
    },
    {
        "label": "No type on root",
        "bad":  '{"properties": {...}}  # no "type": "object"',
        "breaks": "Provider rejects the tool registration at startup.",
        "fix":  'Root must be {"type": "object", "properties": {...}, "required": [...]}',
    },
    {
        "label": "Free-form-anything parameter",
        "bad":  '{"properties": {"input": {"type": "string"}}}',
        "breaks": "Reverts to S29-style — model invents its own micro-format inside the string.",
        "fix":  "Break the string into named typed fields, one per parameter.",
    },
    {
        "label": "Overlapping descriptions",
        "bad":  'tool A: "Look up info."  + tool B: "Find information."',
        "breaks": "LLM picks whichever was listed first. Selection is random.",
        "fix":  "Apply the CONTRAST pattern — each description names what the OTHER doesn't do.",
    },
    {
        "label": "Description equals name",
        "bad":  'name="calculator", description="Calculator."',
        "breaks": "Description carries zero extra signal beyond the name.",
        "fix":  "Three sentences: WHAT, WHEN, EXAMPLE.",
    },
    {
        "label": "Returns a non-string",
        "bad":  'def run(args): return {"answer": 42}  # dict, not string',
        "breaks": "Provider rejects the next turn — tool_result.content must be a string.",
        "fix":  "Format inside run(): json.dumps(result), or build a readable summary.",
    },
]


def render_anti_patterns_as_text() -> str:
    """Pretty-print the anti-pattern gallery for the in-session quick reference."""
    lines = []
    for i, ap in enumerate(ANTI_PATTERNS, start=1):
        lines.append(f"\n── ANTI-PATTERN {i}: {ap['label']} ──")
        lines.append(f"  BAD:    {ap['bad']}")
        lines.append(f"  BREAKS: {ap['breaks']}")
        lines.append(f"  FIX:    {ap['fix']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 72)
    print("Session 30 — The 5 Tool Description Patterns")
    print("=" * 72)
    print(render_patterns_as_text())
    print()
    print("=" * 72)
    print("Session 30 — Anti-Patterns Gallery (8 schema sins)")
    print("=" * 72)
    print(render_anti_patterns_as_text())
