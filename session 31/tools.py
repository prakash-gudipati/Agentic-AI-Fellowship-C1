"""
Session 31 — tools.py

The tool catalog the planner sees, and the executor calls.

This is intentionally a minimal subset of the S30 tool surface — three
deterministic stub tools that any teacher can run live without an internet
connection or a search API key.

The teaching point of S31 is the PLAN, not the tools. Keeping the tools
small means every minute of the session lands on the planner / executor /
monitor / replan loop instead of debugging a search API.

Why these three tools:
  - calculator      — proves the executor can chain numeric work.
  - web_search      — has a small canned corpus including BOTH India and
                      US GDP + population figures for 2024, so the demo
                      "compare per capita" goal can succeed end to end.
  - wikipedia_summary — a second lookup tool, deliberately separate from
                      web_search so the planner has to PICK between them.
                      The contrast also gives us a replan trigger: the
                      planner sometimes routes to the wrong lookup tool
                      and the monitor catches it.

Each tool here returns either a fact (when the query hits the canned
index) OR a structured error string starting with "ERROR:". The executor
treats "ERROR:" prefixes as failures regardless of which tool produced
them — that uniform contract is what makes the monitor cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


# ----------------------------------------------------------------------------
# Tool data type
# ----------------------------------------------------------------------------


@dataclass
class Tool:
    """
    Minimal tool record.

    name        : how the planner refers to this tool in step.tool.
    description : the LLM-facing description the planner reads when
                  deciding which tool to put in each step.
    run         : Python callable. Takes the args dict, returns a string.
    """

    name: str
    description: str
    run: Callable[[Dict[str, Any]], str]


# ----------------------------------------------------------------------------
# Tool 1 — calculator
# ----------------------------------------------------------------------------


def _safe_eval(expression: str) -> str:
    """Same disciplined safe-eval pattern S29 introduced: allowlist chars."""

    allowed = set("0123456789+-*/(). eE")
    bad = [c for c in expression if c not in allowed]
    if bad:
        return (
            "ERROR: calculator only accepts digits and + - * / ( ) . "
            f"Got disallowed characters: {sorted(set(bad))}"
        )
    try:
        # eval is safe here because we've allow-listed the character set.
        value = eval(expression, {"__builtins__": {}}, {})
        return str(value)
    except Exception as exc:
        return f"ERROR: calculator could not evaluate {expression!r} ({exc})"


def _run_calculator(args: Dict[str, Any]) -> str:
    expr = args.get("expression", "")
    if not expr:
        return "ERROR: calculator was called with no expression."
    return _safe_eval(str(expr))


calculator = Tool(
    name="calculator",
    description=(
        "Evaluate a pure-arithmetic expression. Use this whenever the "
        "step needs a number computed from other numbers — sums, ratios, "
        "percentages, divisions. Args: {'expression': str}. Only digits "
        "and + - * / ( ) . are accepted. Returns the number as a string, "
        "or an ERROR: line if the expression is invalid."
    ),
    run=_run_calculator,
)


# ----------------------------------------------------------------------------
# Tool 2 — web_search (canned)
# ----------------------------------------------------------------------------


# Each entry is (required_token_groups, answer_string).
# A query matches when, for every group in required_token_groups, at least
# one token from that group appears in the query. This handles phrasing
# variants like "GDP of India 2024", "India GDP 2024", "India's nominal
# GDP for the year 2024" — they all hit the same fact.
#
# Token groups for the US row include {"united", "states", "us", "usa",
# "america"} so any of those aliases trips the match.
# IMPORTANT: order matters — first-match-wins. More specific entries
# (e.g. "per capita GDP") MUST come before more general ones ("GDP").
_FAKE_SEARCH_INDEX: List[tuple] = [
    # India — per-capita GDP 2024 (most specific — comes first)
    (
        [{"per-capita", "per", "capita", "percapita"}, {"gdp"},
         {"india", "indian"}, {"2024"}],
        "World Bank: India's per-capita GDP for 2024 is approximately "
        "USD 2,731.",
    ),
    # United States — per-capita GDP 2024 (most specific — comes first)
    (
        [{"per-capita", "per", "capita", "percapita"}, {"gdp"},
         {"united", "us", "usa", "america", "american"}, {"2024"}],
        "World Bank: United States per-capita GDP for 2024 is approximately "
        "USD 85,694.",
    ),
    # India — nominal GDP 2024
    (
        [{"gdp"}, {"india", "indian"}, {"2024"}],
        "World Bank: India's nominal GDP for 2024 is approximately "
        "USD 3,937,011,000,000 (3.937 trillion).",
    ),
    # India — population 2024
    (
        [{"population"}, {"india", "indian"}, {"2024"}],
        "World Bank: India's population in 2024 is approximately "
        "1,441,719,852 people.",
    ),
    # United States — nominal GDP 2024
    (
        [{"gdp"}, {"united", "us", "usa", "america", "american"}, {"2024"}],
        "World Bank: United States nominal GDP for 2024 is approximately "
        "USD 28,781,083,000,000 (28.78 trillion).",
    ),
    # United States — population 2024
    (
        [{"population"}, {"united", "us", "usa", "america", "american"}, {"2024"}],
        "World Bank: United States population in 2024 is approximately "
        "335,893,238 people.",
    ),
    # ISRO Mars Orbiter Mission launch date (any of mars/mangalyaan/isro hits)
    (
        [{"isro", "mangalyaan", "mars"}, {"launch", "launched", "date"}],
        "ISRO launched the Mars Orbiter Mission (Mangalyaan) on "
        "5 November 2013 from Sriharikota.",
    ),
]


def _tokenise(text: str) -> set:
    """Lowercase the text and split on non-alphanumeric. Returns a set."""

    import re
    return set(t for t in re.split(r"[^a-z0-9-]+", text.lower()) if t)


def _run_web_search(args: Dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "ERROR: web_search was called with no query."

    query_tokens = _tokenise(query)
    for token_groups, answer in _FAKE_SEARCH_INDEX:
        if all(any(t in query_tokens for t in group) for group in token_groups):
            return answer

    # Prefix with ERROR: so the executor classifies this as a failure and
    # the monitor sees a clear signal. "Found nothing" is a failure mode
    # for the agent even if it isn't a Python exception.
    return (
        f"ERROR: web_search found no high-confidence result for {query!r}. "
        "Try a more specific query or use the wikipedia_summary tool "
        "if you are looking for a definition rather than a current statistic."
    )


web_search = Tool(
    name="web_search",
    description=(
        "Search the web for a current FACT or STATISTIC — GDP, population, "
        "stock price, launch date, etc. Args: {'query': str}. Returns one "
        "sentence with the answer and the source, or a 'No high-confidence "
        "result' line if the query is too vague. Use this for numeric or "
        "time-sensitive facts; use wikipedia_summary for definitions."
    ),
    run=_run_web_search,
)


# ----------------------------------------------------------------------------
# Tool 3 — wikipedia_summary (canned)
# ----------------------------------------------------------------------------


_FAKE_WIKI_INDEX: List[tuple] = [
    (
        [{"alan"}, {"turing"}],
        "Alan Mathison Turing (1912-1954) was a British mathematician "
        "and computer scientist, widely considered the father of "
        "theoretical computer science and artificial intelligence.",
    ),
    (
        [{"react"}, {"pattern", "agent", "loop"}],
        "ReAct (Reason + Act) is an LLM agent pattern in which the model "
        "alternates Thought and Action steps within a single loop, using "
        "the observation from each action to inform the next thought.",
    ),
    (
        [{"mars", "orbiter", "mangalyaan", "mission"}],
        "The Mars Orbiter Mission (Mangalyaan) is India's first "
        "interplanetary mission, launched by ISRO in 2013, which made "
        "India the first nation to reach Mars orbit on its first attempt.",
    ),
]


def _run_wikipedia_summary(args: Dict[str, Any]) -> str:
    topic = str(args.get("topic", "")).strip()
    if not topic:
        return "ERROR: wikipedia_summary was called with no topic."

    topic_tokens = _tokenise(topic)
    for token_groups, answer in _FAKE_WIKI_INDEX:
        if all(any(t in topic_tokens for t in group) for group in token_groups):
            return answer

    return (
        f"ERROR: wikipedia_summary has no article for {topic!r}. "
        "Try web_search if you are looking for a specific statistic "
        "rather than a general definition."
    )


wikipedia_summary = Tool(
    name="wikipedia_summary",
    description=(
        "Return a short summary of a person, concept, or named entity. "
        "Args: {'topic': str}. Use this for DEFINITIONS and background, "
        "not for current statistics. Returns one paragraph from a static "
        "Wikipedia-style index, or a 'no article found' line if absent."
    ),
    run=_run_wikipedia_summary,
)


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------


REGISTRY: Dict[str, Tool] = {
    calculator.name: calculator,
    web_search.name: web_search,
    wikipedia_summary.name: wikipedia_summary,
}


def tool_names() -> List[str]:
    """Sorted list of all tool names. Used by the planner validator."""
    return sorted(REGISTRY.keys())


def render_catalog() -> str:
    """
    Build the text catalog the planner reads in its system prompt.

    The format is deliberately compact:

        - tool_name : one-line description
          args: {"field": "type"}

    The planner does not need JSON Schema — it needs to know which tool
    to pick and what fields to fill in. The validator handles the typing
    later.
    """

    lines = []
    for name in tool_names():
        tool = REGISTRY[name]
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)
