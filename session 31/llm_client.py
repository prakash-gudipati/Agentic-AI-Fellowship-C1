"""
Session 31 — llm_client.py

A deliberately thin Anthropic wrapper.

S30 introduced a provider abstraction so the agent loop could swap between
Anthropic and OpenAI. In S31 we keep it simple — one provider, one model,
two helper methods — because the topic of the session is PLANNING, not
provider portability. Students who want to swap providers re-read S30 §
"Provider Adapters" and port it.

What this file exposes:
  - LLMClient.complete(system, user, max_tokens)  -> str
      a one-shot text completion. Used by the planner, the monitor, and
      the synthesis step.
  - LLMClient.complete_react_turn(system, messages, tools, max_tokens) -> ToolUseTurn
      a single native-function-calling turn. Used by react_baseline.py
      so we can compare ReAct against Plan-and-Execute on equal footing.

The "fake mode" path returns deterministic offline replies so the demos
can run in classrooms with flaky internet. The classroom instructor
controls fake mode via the FAKE_LLM env var (set to "1" to enable).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------------


@dataclass
class ToolUseTurn:
    """
    One turn from the ReAct baseline.

    stop_reason values:
      "end_turn"  — the model produced a text-only answer; loop exits.
      "tool_use" — the model asked to call one or more tools.
      "max_tokens" — token budget exhausted (should be rare).

    tool_calls is a list of (tool_name, args, tool_use_id) for the executor
    to action and feed back as tool_result blocks. tool_use_id is the
    string Anthropic assigns to each tool call — the same id must come
    back in the tool_result block or the API rejects the next call.
    """

    text: str
    stop_reason: str
    tool_calls: List[Dict[str, Any]]


# ----------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------


class LLMClient:
    """
    A thin wrapper around the Anthropic Python SDK.

    Public surface intentionally small. Two methods. No retries, no
    streaming, no caching — the production patterns S26 (LangSmith) and
    S30 (action log) live outside this client.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        self.model = model
        self._fake = os.environ.get("FAKE_LLM", "") == "1"
        self._client = None  # lazily constructed so the demo can import
                            # this file without anthropic installed

    # ------------------------------------------------------------------
    # Text completion (used by planner + monitor + synthesis)
    # ------------------------------------------------------------------

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """
        One-shot text completion. Returns the response text.

        The system+user format mirrors the way the Anthropic API splits
        roles. If FAKE_LLM=1, returns a canned response keyed by a hash
        of the inputs — kept deterministic so demos are reproducible.
        """

        if self._fake:
            return _fake_text_completion(system, user)

        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate any text blocks; ignore tool_use blocks (this entry
        # point is text-only).
        out_parts: List[str] = []
        for block in message.content:
            if getattr(block, "type", "") == "text":
                out_parts.append(block.text)
        return "".join(out_parts).strip()

    # ------------------------------------------------------------------
    # Native function-calling turn (used by react_baseline.py)
    # ------------------------------------------------------------------

    def complete_react_turn(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 1024,
    ) -> ToolUseTurn:
        """
        Run one round-trip of the ReAct conversation.

        - system     : the agent's standing instructions.
        - messages   : the conversation so far, in Anthropic message format.
                       Tool results are passed as user-role tool_result
                       blocks. The caller appends each round.
        - tools      : Anthropic tool definitions (name, description,
                       input_schema). The same shape S30 uses.

        Returns a ToolUseTurn the caller can inspect. If stop_reason is
        "tool_use", the caller runs each tool and appends a user
        message with the tool_result blocks before calling again.
        """

        if self._fake:
            return _fake_react_turn(system, messages, tools)

        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in message.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "args": dict(block.input or {}),
                    }
                )
        return ToolUseTurn(
            text="".join(text_parts).strip(),
            stop_reason=message.stop_reason or "end_turn",
            tool_calls=tool_calls,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic SDK not installed. Run 'pip install anthropic' "
                    "or set FAKE_LLM=1 to use the offline demo path."
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client


# ----------------------------------------------------------------------------
# Offline fake responses
# ----------------------------------------------------------------------------


def _fake_text_completion(system: str, user: str) -> str:
    """
    Deterministic offline responses for the planner / monitor / synthesis.

    The fakes are keyed by a substring match against the USER prompt. This
    is enough to demo Plan-and-Execute end to end without an API key.
    """

    lowered = user.lower().replace("-", " ")

    # Planner — the demo goal asks to compare per capita GDP India vs US.
    if "produce a plan as json" in lowered and "per capita" in lowered:
        plan = {
            "synthesis_hint": (
                "Compare India's per-capita GDP and the US's per-capita GDP "
                "for 2024 and state which is larger plus the ratio."
            ),
            "steps": [
                {"id": "s1", "description": "Look up India GDP 2024",
                 "tool": "web_search",
                 "args": {"query": "gdp of india 2024"}, "depends_on": []},
                {"id": "s2", "description": "Look up India population 2024",
                 "tool": "web_search",
                 "args": {"query": "population of india 2024"},
                 "depends_on": []},
                {"id": "s3", "description": "Look up United States GDP 2024",
                 "tool": "web_search",
                 "args": {"query": "gdp of united states 2024"},
                 "depends_on": []},
                {"id": "s4", "description": "Look up United States population 2024",
                 "tool": "web_search",
                 "args": {"query": "population of united states 2024"},
                 "depends_on": []},
                {"id": "s5",
                 "description": "Compute India per-capita GDP",
                 "tool": "calculator",
                 "args": {"expression": "3937011000000 / 1441719852"},
                 "depends_on": ["s1", "s2"]},
                {"id": "s6",
                 "description": "Compute United States per-capita GDP",
                 "tool": "calculator",
                 "args": {"expression": "28781083000000 / 335893238"},
                 "depends_on": ["s3", "s4"]},
                {"id": "s7",
                 "description": "Compute ratio US-per-capita / India-per-capita",
                 "tool": "calculator",
                 "args": {"expression": "(28781083000000 / 335893238) / (3937011000000 / 1441719852)"},
                 "depends_on": ["s5", "s6"]},
            ],
        }
        return json.dumps(plan)

    # Planner — drift demo goal asks about an unknown lookup.
    if "produce a plan as json" in lowered and "alan turing" in lowered:
        plan = {
            "synthesis_hint": "Summarise who Alan Turing was in two sentences.",
            "steps": [
                # Deliberately wrong first attempt — points web_search at a
                # biography query that the canned index does NOT have. This
                # produces the drift the monitor catches.
                {"id": "s1",
                 "description": "Look up Alan Turing biography",
                 "tool": "web_search",
                 "args": {"query": "alan turing biography"},
                 "depends_on": []},
            ],
        }
        return json.dumps(plan)

    # Replan — after a drift trigger on the Alan Turing goal.
    if "revise the plan" in lowered and "alan turing" in lowered:
        plan = {
            "synthesis_hint": "Summarise who Alan Turing was in two sentences.",
            "steps": [
                {"id": "s1",
                 "description": "Look up Alan Turing on Wikipedia",
                 "tool": "wikipedia_summary",
                 "args": {"topic": "alan turing"},
                 "depends_on": []},
            ],
        }
        return json.dumps(plan)

    # Monitor — returns a structured drift decision.
    if "you are the monitor" in lowered:
        if "error" in lowered or "no high-confidence" in lowered or "no wikipedia article" in lowered:
            return json.dumps({
                "should_replan": True,
                "reason": "Last step returned an error or empty result.",
            })
        return json.dumps({"should_replan": False, "reason": "ok"})

    # Synthesis — produce a short final answer that names the per-capita gap.
    if "synthesise the final answer" in lowered and "per capita" in lowered:
        return (
            "India's per-capita GDP in 2024 was about USD 2,731 while the "
            "United States stood at about USD 85,686 — the US figure is "
            "roughly 31.4 times larger."
        )

    if "synthesise the final answer" in lowered and "alan turing" in lowered:
        return (
            "Alan Turing (1912-1954) was a British mathematician widely "
            "regarded as the father of theoretical computer science and "
            "artificial intelligence."
        )

    # Fall-through default — keeps the demo from crashing on stray prompts.
    return "[fake LLM] no canned response matched; returning empty."


def _fake_react_turn(
    system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
) -> ToolUseTurn:
    """
    Deterministic ReAct turns for the comparison demo.

    Counts how many tool_result blocks have already been delivered to
    derive the next tool call. This is intentionally simple — the goal
    is to show ReAct chaining 7 tool calls to do the same work the planner
    did in one plan + execution.
    """

    delivered = _count_tool_results(messages)

    sequences = [
        ("web_search", {"query": "gdp of india 2024"}, "Look up India GDP."),
        ("web_search", {"query": "population of india 2024"},
         "Look up India population."),
        ("calculator",
         {"expression": "3937011000000 / 1441719852"},
         "Compute India per-capita GDP."),
        ("web_search", {"query": "gdp of united states 2024"},
         "Look up US GDP."),
        ("web_search", {"query": "population of united states 2024"},
         "Look up US population."),
        ("calculator",
         {"expression": "28781083000000 / 335893238"},
         "Compute US per-capita GDP."),
        ("calculator",
         {"expression": "(28781083000000 / 335893238) / (3937011000000 / 1441719852)"},
         "Compute the ratio."),
    ]

    if delivered < len(sequences):
        name, args, thought = sequences[delivered]
        return ToolUseTurn(
            text=f"Thought: {thought}",
            stop_reason="tool_use",
            tool_calls=[{
                "id": f"toolu_{delivered+1:04d}",
                "name": name,
                "args": args,
            }],
        )

    # All seven calls done — emit the final synthesis as text and stop.
    return ToolUseTurn(
        text=(
            "India's per-capita GDP in 2024 was about USD 2,731 while the "
            "United States stood at about USD 85,686 — the US figure is "
            "roughly 31.4 times larger."
        ),
        stop_reason="end_turn",
        tool_calls=[],
    )


def _count_tool_results(messages: List[Dict[str, Any]]) -> int:
    """How many tool_result blocks have been delivered back to the model."""

    count = 0
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    count += 1
    return count
