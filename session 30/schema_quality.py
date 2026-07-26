"""
Session 30 - schema_quality.py

The single most-replayed teaching moment of the session: VAGUE vs SHARP
tool schemas, side by side, on the same model, with the same questions.

Why does this exist as its own module?
  - The point of Session 30 is not "we have tools now". You had tools in
    Session 29. The point is that *the description and the schema are how
    the LLM picks the right tool*. We can show that empirically only if we
    can run the same agent twice - once with bad descriptions, once with
    good ones - and compare what it picks.
  - Keeping the two registries side by side in one file is the visual:
    the SAME three tools, the SAME run() functions, but DIFFERENT
    descriptions and DIFFERENT JSON schemas. The only thing that changes
    is what the LLM sees. That's the whole lesson.

What the demo will show:
  - VAGUE: opaque tool names (tool_a / tool_b / tool_c), identical generic
    descriptions ("Returns a result."), anonymous parameter `x`. The model
    has literally zero signal to differentiate. Expect non-deterministic
    selection and wasted turns.
  - SHARP: domain-named tools with three-sentence descriptions and typed
    parameters. The model picks correctly on the first try.

v3 FIX A (May 2026): the earlier vague names (compute / lookup / thing)
gave modern Haiku-class models too much signal - they routed correctly
from the name alone, undermining the contrast. Renaming to tool_a/b/c
with identical generic descriptions removes ALL signal, forcing the
contrast to be visible in the live demo.

We do NOT change the underlying tool implementations. The VAGUE versions
call the same _run_calculator, _run_web_search, _run_wikipedia_summary
functions. That is the controlled experiment: the only variable is the
metadata the LLM reads.
"""

from __future__ import annotations

from typing import List

from tools import (
    Tool,
    _run_calculator,
    _run_web_search,
    _run_wikipedia_summary,
    calculator_tool,
    web_search_tool,
    wikipedia_summary_tool,
)


# ── VAGUE tools - opaque names, identical generic descriptions ──────────────
# All three tools have the SAME description and an anonymous parameter `x`.
# The model has nothing to differentiate them by. Selection is guesswork.

GENERIC_DESCRIPTION = "Returns a result."

vague_tool_a = Tool(
    name="tool_a",
    description=GENERIC_DESCRIPTION,
    input_schema={
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    },
    run=lambda args: _run_calculator({"expression": args.get("x", "")}),
)

vague_tool_b = Tool(
    name="tool_b",
    description=GENERIC_DESCRIPTION,
    input_schema={
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    },
    run=lambda args: _run_web_search({"query": args.get("x", "")}),
)

vague_tool_c = Tool(
    name="tool_c",
    description=GENERIC_DESCRIPTION,
    input_schema={
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    },
    run=lambda args: _run_wikipedia_summary({"topic": args.get("x", "")}),
)

VAGUE_TOOLS: List[Tool] = [vague_tool_a, vague_tool_b, vague_tool_c]

# Re-export the sharp set for symmetry with VAGUE_TOOLS.
SHARP_TOOLS: List[Tool] = [calculator_tool, web_search_tool, wikipedia_summary_tool]
