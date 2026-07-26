"""
Session 31 — react_baseline.py

A minimal ReAct loop, here only so the demo can put ReAct and
Plan-and-Execute side by side on the same goal.

This is NOT a polished agent. Refer back to S29 (`Session_29/Code/`) for
the pedagogical ReAct walkthrough and to S30 (`Session_30/Code/`) for the
production-grade native-function-calling version.

What this file gives you:
  - A simple `run_react(goal)` that uses Anthropic native function calling
    in a loop, accumulates the conversation, and returns:
        text             — the final answer
        tool_call_count  — how many tool calls were made
        turns            — how many model round-trips happened
  - Counters so demo.py can print "Plan-and-Execute took 1 planner call +
    7 tool calls + 1 synthesis call; ReAct took 8 model round-trips".

The fake-mode path mirrors what Claude does on the demo per-capita-GDP
goal — chains 7 tool calls before answering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from llm_client import LLMClient, ToolUseTurn
from tools import REGISTRY


REACT_SYSTEM_PROMPT = """You are a tool-using assistant.

When you need to compute a number, look up a fact, or summarise a topic,
call one of the tools. Reason briefly between tool calls. When you have
enough information, answer the user directly without calling any more tools.
"""


@dataclass
class ReactRun:
    """What a single ReAct execution recorded."""

    text: str
    tool_call_count: int
    turns: int


def run_react(goal: str, max_turns: int = 15) -> ReactRun:
    """
    Run a minimal ReAct loop. Returns a ReactRun summary.

    The loop:
      - call the model with the current conversation
      - if the model emits tool_use blocks, run each tool and append the
        results back to the conversation as tool_result blocks
      - if the model emits text only, return
      - stop at max_turns even if no end_turn arrived
    """

    client = LLMClient()
    tools = _anthropic_tool_schemas()

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": goal}
    ]
    tool_call_count = 0

    for turn_index in range(max_turns):
        turn = client.complete_react_turn(
            system=REACT_SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
        )

        # Append assistant message — combine text + tool_use blocks in order.
        assistant_blocks: List[Dict[str, Any]] = []
        if turn.text:
            assistant_blocks.append({"type": "text", "text": turn.text})
        for call in turn.tool_calls:
            assistant_blocks.append({
                "type": "tool_use",
                "id": call["id"],
                "name": call["name"],
                "input": call["args"],
            })
        messages.append({"role": "assistant", "content": assistant_blocks})

        if turn.stop_reason == "tool_use" and turn.tool_calls:
            tool_results: List[Dict[str, Any]] = []
            for call in turn.tool_calls:
                tool_call_count += 1
                tool = REGISTRY.get(call["name"])
                if tool is None:
                    obs = f"ERROR: unknown tool {call['name']!r}."
                else:
                    try:
                        obs = tool.run(call["args"])
                    except Exception as exc:
                        obs = f"ERROR: tool raised {type(exc).__name__}: {exc}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": obs,
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        # No tool_use this turn => model emitted a final answer.
        return ReactRun(
            text=turn.text or "(no text)",
            tool_call_count=tool_call_count,
            turns=turn_index + 1,
        )

    return ReactRun(
        text="(loop hit max_turns without an end_turn)",
        tool_call_count=tool_call_count,
        turns=max_turns,
    )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _anthropic_tool_schemas() -> List[Dict[str, Any]]:
    """
    Translate the S31 tool registry into Anthropic tool-definition shape.

    Inlined here (instead of importing from S30) so this file stays a
    standalone reference. If you want the production-grade provider
    abstraction, re-read S30's providers/anthropic.py.
    """

    return [
        {
            "name": "calculator",
            "description": REGISTRY["calculator"].description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "An arithmetic expression using digits and "
                            "+ - * / ( ) . only."
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "web_search",
            "description": REGISTRY["web_search"].description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short query for a current fact.",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "wikipedia_summary",
            "description": REGISTRY["wikipedia_summary"].description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Name of the person, concept, or entity.",
                    }
                },
                "required": ["topic"],
            },
        },
    ]
