"""
Session 29 — tools.py

The tools an agent can reach for.

Why does this exist as its own module?
  - The agent in agent.py is generic — it doesn't know what a calculator is,
    it doesn't know what a clock is. It only knows "tools have a name, a
    description, and a run() method." That separation is what lets us add
    or swap tools without touching the agent loop.
  - In production, your tool registry is where governance lives: which tools
    are exposed to which agent, with what input contracts, with what
    rate limits. Keeping it in one place pays off forever.

Each tool is small on purpose. The point of Session 29 is the LOOP, not the
tool implementation. We use a hardcoded web-search stub so the demo runs
deterministically in any classroom — no API key needed for search, no
flakiness from upstream rate limits.

Production patterns (S29 first build, Phase 5):
  - Single-responsibility tool classes                       (S3)
  - Named constants at file top                              (S4)
  - Try/except around every external/eval boundary           (S3)
  - Long descriptive names — no abbreviations                (S7)
  - Tool description is part of the contract the LLM reads   (S29 new)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict


# ── Safe-eval allowlist for the calculator ──────────────────────────────────
# We refuse to call Python's eval() on whatever the LLM emits. Even though
# this is a teaching project, the habit of refusing unsafe input is the
# habit we want students to leave with.
ALLOWED_CALCULATOR_CHARS = set("0123456789+-*/(). ")


@dataclass
class Tool:
    """A tool the agent can call.

    name        : short identifier the LLM will emit in `Action: <name>[...]`
    description : one-line description the LLM sees when deciding which tool
                  to use. This is the LLM-facing contract — write it like
                  you would write a job description for a specialist.
    run         : a function that takes a raw string input from the agent
                  and returns a string observation.
    """
    name: str
    description: str
    run: Callable[[str], str]


# ── Tool 1: Calculator ──────────────────────────────────────────────────────

def _run_calculator(raw_input: str) -> str:
    """Evaluate a simple arithmetic expression. Safe-eval only."""
    expression = raw_input.strip()
    if not expression:
        return "ERROR: calculator received an empty expression."

    # Refuse anything that isn't pure arithmetic.
    illegal = [ch for ch in expression if ch not in ALLOWED_CALCULATOR_CHARS]
    if illegal:
        return (
            f"ERROR: calculator only accepts digits and + - * / ( ) . "
            f"Got disallowed characters: {sorted(set(illegal))}"
        )

    try:
        # eval is acceptable HERE because we've allowlisted every character.
        # In real production code we would use a proper expression parser.
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as exc:                                # noqa: BLE001
        return f"ERROR: calculator could not evaluate '{expression}' — {exc}"

    return f"{result}"


calculator_tool = Tool(
    name="calculator",
    description=(
        "Evaluate a single arithmetic expression. "
        "Input must be a math expression using only digits and + - * / ( ). "
        "Example input: 1452749200 / 1428627663"
    ),
    run=_run_calculator,
)


# ── Tool 2: Current date / time ─────────────────────────────────────────────

def _run_datetime(_raw_input: str) -> str:
    """Return the current UTC date and time. Ignores any input."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M UTC")


datetime_tool = Tool(
    name="current_datetime",
    description=(
        "Return the current UTC date and time. Takes no input — pass an empty string. "
        "Use this when the question depends on what 'today' is."
    ),
    run=_run_datetime,
)


# ── Tool 3: Web-search stub ─────────────────────────────────────────────────
# A real web-search tool would call SerpAPI / Tavily / Brave. For Session 29
# we use a hardcoded lookup table so the demo runs deterministically in any
# classroom, even with no internet. In S30 students will swap this stub for
# a real search API.

_FAKE_SEARCH_INDEX: Dict[str, str] = {
    "gdp of india 2024": (
        "World Bank: India's nominal GDP for 2024 is approximately "
        "USD 3,937,011,000,000 (3.937 trillion)."
    ),
    "population of india 2024": (
        "UN Population Division: India's estimated population in 2024 is "
        "approximately 1,441,719,852 people."
    ),
    "gdp of united states 2024": (
        "World Bank: The United States' nominal GDP for 2024 is approximately "
        "USD 28,781,000,000,000 (28.78 trillion)."
    ),
    "population of united states 2024": (
        "US Census Bureau: The United States population on 1 January 2024 was "
        "approximately 335,893,238 people."
    ),
}


def _run_web_search(raw_input: str) -> str:
    """A stand-in for a real web search. Returns canned snippets."""
    query = raw_input.strip().lower().strip("\"'")
    if not query:
        return "ERROR: web_search received an empty query."

    # Naive substring match — good enough for the demo.
    for key, snippet in _FAKE_SEARCH_INDEX.items():
        if all(word in query for word in key.split()):
            return snippet

    return (
        f"No high-confidence result for '{raw_input.strip()}'. "
        f"Try a more specific query or a different tool."
    )


web_search_tool = Tool(
    name="web_search",
    description=(
        "Look up a single fact on the public web. "
        "Input should be a short search query as a string. "
        "Example input: GDP of India 2024"
    ),
    run=_run_web_search,
)


# ── Registry the agent reads at startup ─────────────────────────────────────
TOOL_REGISTRY: Dict[str, Tool] = {
    calculator_tool.name: calculator_tool,
    datetime_tool.name: datetime_tool,
    web_search_tool.name: web_search_tool,
}


def render_tool_catalog() -> str:
    """Render the tool list into the format the system prompt expects."""
    lines = []
    for tool in TOOL_REGISTRY.values():
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)
