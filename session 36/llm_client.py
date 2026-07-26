"""
Session 36 — llm_client.py

Thin Anthropic wrapper with a FAKE_LLM=1 offline mode.

What is REAL vs FAKE in this session matters:

  * The MCP transport is ALWAYS real. The server subprocess actually runs,
    the client actually performs the discovery handshake, and tools are
    actually called over stdio. Nothing about MCP is mocked.
  * Only the agent's BRAIN (which tool to call next) is faked when
    FAKE_LLM=1, so the demos are deterministic and need no API key.

PHASE 5 RULE — FAKE_LLM route matching uses opener phrases, never topic words.
Here there is a single agent prompt ("You are an agent ..."), so the fake
decision logic keys off the user question + tool-call history instead.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class AgentDecision:
    kind: str  # "tool_call" or "final"
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    text: str = ""
    raw_assistant_blocks: List[Dict[str, Any]] = field(default_factory=list)
    tool_use_id: str = ""


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._fake = os.environ.get("FAKE_LLM", "") == "1"
        self._client = None
        self.llm_calls = 0

    def decide_next(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 800,
    ) -> AgentDecision:
        """One agent step: call a tool, or write the final answer."""

        self.llm_calls += 1
        if self._fake:
            return _fake_decide(messages)

        client = self._get_client()
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            # Our loop executes ONE tool per step, so ask the model not to emit
            # parallel tool calls. Without this, the model can return write_file
            # AND read_file in a single turn; we'd answer only one, leaving the
            # other tool_use without a matching tool_result and the next API
            # call would 400 with "tool_use ids found without tool_result".
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=messages,
        )
        text_blocks: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        first_tool: Dict[str, Any] = {}
        for block in msg.content:
            t = getattr(block, "type", "")
            if t == "text":
                text_parts.append(block.text)
                text_blocks.append({"type": "text", "text": block.text})
            elif t == "tool_use" and not first_tool:
                first_tool = {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            # Any further tool_use blocks are deliberately dropped — the
            # assistant turn we record must contain exactly the one tool_use we
            # are about to answer, so every tool_use has a matching tool_result.
        if first_tool:
            return AgentDecision(
                kind="tool_call",
                tool_name=first_tool["name"],
                tool_args=first_tool.get("input", {}) or {},
                text="".join(text_parts).strip(),
                raw_assistant_blocks=text_blocks + [first_tool],
                tool_use_id=first_tool["id"],
            )
        return AgentDecision(
            kind="final",
            text="".join(text_parts).strip(),
            raw_assistant_blocks=text_blocks,
        )

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic SDK not installed. Run 'pip install anthropic' "
                    "or set FAKE_LLM=1."
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client


# ============================================================================
# FAKE_LLM agent brain — deterministic decisions for the demos.
# ============================================================================


def _fake_decide(messages: List[Dict[str, Any]]) -> AgentDecision:
    first_user = ""
    calls_made: List[str] = []
    last_observation = ""

    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "user" and isinstance(content, str):
            if not first_user:
                first_user = content
        elif role == "user" and isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "tool_result":
                    last_observation = str(blk.get("content", ""))
        elif role == "assistant" and isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "tool_use":
                    calls_made.append(blk.get("name", ""))

    q = first_user.strip()
    q_l = q.lower()
    n = len(calls_made)

    # --- write-then-read flow (Demo 3) ------------------------------------
    wants_write = any(w in q_l for w in ("create a file", "write a file", "save"))
    if wants_write and "write_file" not in calls_made:
        path = _parse_filename(q) or "summary.txt"
        body = _parse_quoted(q) or "MCP standardises tool access."
        return AgentDecision(
            kind="tool_call",
            tool_name="write_file",
            tool_args={"path": path, "content": body},
            tool_use_id=f"call_{n+1}",
        )
    if wants_write and "read it back" in q_l and "read_file" not in calls_made:
        path = _parse_filename(q) or "summary.txt"
        return AgentDecision(
            kind="tool_call",
            tool_name="read_file",
            tool_args={"path": path},
            tool_use_id=f"call_{n+1}",
        )

    # --- read-a-file flow (Demo 2) ----------------------------------------
    wants_read = ("read" in q_l or ".txt" in q_l) and not wants_write
    if wants_read and "read_file" not in calls_made:
        path = _parse_filename(q) or "product_brief.txt"
        return AgentDecision(
            kind="tool_call",
            tool_name="read_file",
            tool_args={"path": path},
            tool_use_id=f"call_{n+1}",
        )

    # --- web-search flow (Demo 4) -----------------------------------------
    wants_search = any(w in q_l for w in ("search the web", "look up", "search for"))
    if wants_search and "search_web" not in calls_made:
        return AgentDecision(
            kind="tool_call",
            tool_name="search_web",
            tool_args={"query": _strip_command(q)},
            tool_use_id=f"call_{n+1}",
        )

    # --- otherwise, synthesise the final answer from the last observation -
    return AgentDecision(kind="final", text=_synthesise(q, last_observation))


def _synthesise(question: str, observation: str) -> str:
    if not observation:
        return "I could not find the information needed to answer that."
    q_l = question.lower()
    # Demo 2 asks for the launch date — pull the matching line if present.
    if "launch" in q_l or "date" in q_l:
        for line in observation.splitlines():
            if "launch" in line.lower() or "date" in line.lower():
                return f"Based on the file, {line.strip()}"
    if observation.startswith("OK: wrote"):
        return f"Done. {observation}"
    # Default: hand back what the tool returned, lightly framed.
    head = observation.strip().splitlines()
    body = "\n".join(head[:6])
    return f"Here is what I found:\n{body}"


def _parse_filename(text: str) -> str:
    m = re.search(r"([A-Za-z0-9_\-/]+\.[A-Za-z0-9]{1,5})", text)
    return m.group(1) if m else ""


def _parse_quoted(text: str) -> str:
    m = re.search(r"['\"]([^'\"]+)['\"]", text)
    return m.group(1) if m else ""


def _strip_command(text: str) -> str:
    # Turn "Search the web for X and summarise" into "X".
    t = re.sub(r"(?i)^(search the web for|look up|search for)\s+", "", text).strip()
    t = re.sub(r"(?i)\s+and summarise.*$", "", t).strip()
    return t or text
