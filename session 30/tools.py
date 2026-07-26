"""
Session 30 — tools.py

The tools the agent can reach for, with their *schemas*.

Why does this exist as its own module?
  - In Session 29 the agent emitted `Action: tool_name[free-text input]` and we
    parsed it with a regex. That worked, but it is fragile in two ways:
        1. the input is a single string, so a tool that needs two fields has
           to invent its own micro-format inside that string;
        2. the LLM only knows about each tool from a free-text catalog we
           stuff into the system prompt — there is no contract the API itself
           validates.
  - Function calling fixes both. Each tool now carries a JSON Schema that
    declares its inputs by name and type. The provider validates the call
    against the schema before it ever reaches our Python.
  - We keep the registry in one file so governance lives in one file —
    "which tools exist, what they accept, what they describe themselves as"
    is the single most important block of code in any agent product.

Each tool has FOUR fields:
  name        : short identifier the LLM emits and the registry keys by.
  description : the LLM-facing "job description". This is the line that
                decides whether a tool gets picked. Treat it like the
                first sentence of a job posting — sharp, specific, with an
                example. (Slide 9 in the deck makes this concrete.)
  input_schema: JSON Schema describing the tool's parameters. This is
                what the provider's tool-calling API validates against.
  run         : a Python callable that takes the *parsed dict* of
                arguments and returns a string observation.

Production patterns (S30, building on S29):
  - Tool description is the LLM-facing contract                  (S29 reinforced)
  - Tool input schema is the API-validated contract              (S30 new)
  - Strict allow-listed safe-eval for the calculator             (S29 reinforced)
  - Deterministic offline stubs for demo reproducibility         (S29 reinforced)
  - Wrapped run() so a Python error becomes a structured string  (S30 new)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List


# ── Safe-eval allowlist for the calculator ──────────────────────────────────
# We refuse to call Python's eval() on whatever the LLM emits. Schemas guard
# the SHAPE; this allowlist guards the CONTENT. Defence in depth.
ALLOWED_CALCULATOR_CHARS = set("0123456789+-*/(). ")


# ── The Tool dataclass — the LLM-facing contract ────────────────────────────
@dataclass
class Tool:
    """A tool the agent can call.

    Fields are intentionally minimal — every extra field is one more thing
    to keep in sync between the schema and the code. Keep it small.

    `idempotent` flag (S30 v3): if False, the retry helper will NOT re-run
    this tool after a failure - it returns the error to the model directly.
    Set False for tools that have side effects (charge card, send email,
    book flight) where double-execution is worse than failure.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]
    run: Callable[[Dict[str, Any]], str]
    idempotent: bool = True


# ── Tool 1: Calculator ──────────────────────────────────────────────────────
# We expose ONE parameter: `expression`. That's the contract.
# The schema is what the model API checks. The body is what we check.

def _run_calculator(args: Dict[str, Any]) -> str:
    """Evaluate a simple arithmetic expression. Safe-eval only."""
    expression = str(args.get("expression", "")).strip()
    if not expression:
        return "ERROR: calculator received an empty expression."

    illegal = [ch for ch in expression if ch not in ALLOWED_CALCULATOR_CHARS]
    if illegal:
        return (
            f"ERROR: calculator only accepts digits and + - * / ( ) . "
            f"Got disallowed characters: {sorted(set(illegal))}"
        )

    try:
        # eval is acceptable HERE because we allowlisted every character.
        # In real production code we would use a proper expression parser.
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as exc:                                # noqa: BLE001
        return f"ERROR: calculator could not evaluate '{expression}' — {exc}"

    return f"{result}"


calculator_tool = Tool(
    name="calculator",
    description=(
        "Evaluate a single arithmetic expression and return the numeric result. "
        "Use this whenever the question requires arithmetic on numbers you already have. "
        "Do NOT use this to look up facts. Example expression: '3937011000000 / 1441719852'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "A single arithmetic expression using only digits and the operators "
                    "+ - * / ( ) and the decimal point. Example: '12 * 7'."
                ),
            }
        },
        "required": ["expression"],
    },
    run=_run_calculator,
)


# ── Tool 2: Web search ──────────────────────────────────────────────────────
# In your portfolio build (the S30 exercise) you swap this stub for a real
# search API — SerpAPI, Tavily, Brave, etc. The schema does not change.
# Demos that don't need the network are demos that finish on time.

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
    "world health organization founded": (
        "Wikipedia: The World Health Organization (WHO) was established on "
        "7 April 1948 and is a specialized agency of the United Nations."
    ),
    "isro mars orbiter mission launch date": (
        "ISRO: The Mars Orbiter Mission (Mangalyaan) was launched on "
        "5 November 2013 from the Satish Dhawan Space Centre."
    ),
}


def _run_web_search(args: Dict[str, Any]) -> str:
    """Return a canned search snippet for the query string."""
    query = str(args.get("query", "")).strip().lower().strip("\"'")
    if not query:
        return "ERROR: web_search received an empty query."

    for key, snippet in _FAKE_SEARCH_INDEX.items():
        if all(word in query for word in key.split()):
            return snippet

    return (
        f"No high-confidence result for '{args.get('query', '').strip()}'. "
        "Try a more specific query or use the wikipedia_summary tool instead."
    )


web_search_tool = Tool(
    name="web_search",
    description=(
        "Look up a single fact on the public web. "
        "Use this when the question depends on information the model could not have memorised — "
        "current statistics, recent events, dated figures. "
        "Return is a one-sentence snippet from the top result. "
        "Example query: 'GDP of India 2024'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A short, specific web-search query. 3-8 words is ideal. "
                    "Example: 'population of India 2024'."
                ),
            }
        },
        "required": ["query"],
    },
    run=_run_web_search,
)


# ── Tool 3: Wikipedia summary ───────────────────────────────────────────────
# Different shape from web_search. Web search returns a snippet from any
# website. Wikipedia returns an encyclopedia-style paragraph for a specific
# named topic — better for biographies and definitions, worse for stats.
# Two tools with overlapping-but-different jobs — the LLM has to read both
# descriptions and pick. This is the whole reason tool descriptions matter.

_FAKE_WIKIPEDIA_INDEX: Dict[str, str] = {
    "alan turing": (
        "Alan Mathison Turing (1912-1954) was a British mathematician and computer scientist, "
        "widely considered the father of theoretical computer science and artificial intelligence. "
        "He proposed the Turing machine in 1936 and led the Hut 8 team at Bletchley Park "
        "that broke the German Enigma cipher during the Second World War."
    ),
    "react pattern": (
        "ReAct (Reasoning + Acting) is a 2022 prompting pattern introduced by Yao et al. "
        "that interleaves model reasoning ('Thought:') with tool use ('Action:') so a "
        "language model can iteratively gather information from external sources before answering."
    ),
    "world health organization": (
        "The World Health Organization (WHO) is a specialized agency of the United Nations "
        "responsible for international public health. Founded on 7 April 1948 and headquartered "
        "in Geneva, Switzerland, it has 194 member states as of 2024."
    ),
    "mars orbiter mission": (
        "The Mars Orbiter Mission (also known as Mangalyaan) was India's first interplanetary "
        "mission, launched by ISRO on 5 November 2013. It successfully entered Martian orbit "
        "on 24 September 2014, making India the first country to reach Mars on its first attempt."
    ),
}


def _run_wikipedia_summary(args: Dict[str, Any]) -> str:
    """Return a canned encyclopedia summary for the requested topic."""
    topic = str(args.get("topic", "")).strip().lower().strip("\"'")
    if not topic:
        return "ERROR: wikipedia_summary received an empty topic."

    for key, summary in _FAKE_WIKIPEDIA_INDEX.items():
        if all(word in topic for word in key.split()):
            return summary

    return (
        f"No Wikipedia article found for '{args.get('topic', '').strip()}'. "
        "Try a different topic name or use web_search if you are looking for a "
        "specific statistic rather than a general definition."
    )


wikipedia_summary_tool = Tool(
    name="wikipedia_summary",
    description=(
        "Return a one-to-two-paragraph encyclopedia-style summary of a named topic, person, "
        "organisation, event, or concept. "
        "Use this when the question asks WHO or WHAT something is, not for live statistics. "
        "Example topic: 'Alan Turing' or 'World Health Organization'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": (
                    "The exact name of the Wikipedia article topic. "
                    "Use the most canonical form of the name. Example: 'Mars Orbiter Mission'."
                ),
            }
        },
        "required": ["topic"],
    },
    run=_run_wikipedia_summary,
)


# ── The registry the agent reads at startup ─────────────────────────────────
SHARP_TOOLS: List[Tool] = [calculator_tool, web_search_tool, wikipedia_summary_tool]


# Convenience lookup the agent loop uses to dispatch a tool_use block.
def build_tool_lookup(tools: List[Tool]) -> Dict[str, Tool]:
    """Return a {name: Tool} map. Single source of truth for dispatch."""
    return {tool.name: tool for tool in tools}


# Datetime helper used only for the action log timestamp — not exposed as a tool.
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()



# ── format_observation - tool output design helper (S30 v3) ────────────────
# Tools return a string. The SHAPE of that string is what the next-turn
# LLM reads and parses. Good shape conventions:
#   - Prefix with a clear status marker: "RESULT:" or "ERROR[category]:"
#   - One line of headline data, then optional detail lines
#   - Keep numbers in canonical form (no comma separators) so a downstream
#     calculator can use them directly
#
# This helper makes the convention reusable across tools.

def format_observation(
    headline: str,
    *,
    detail: str | None = None,
    source: str | None = None,
) -> str:
    """Format a successful tool result in the canonical shape.

    Example output:
        RESULT: 3937011000000
        detail: India nominal GDP 2024
        source: World Bank
    """
    lines = [f"RESULT: {headline}"]
    if detail:
        lines.append(f"detail: {detail}")
    if source:
        lines.append(f"source: {source}")
    return "\n".join(lines)
