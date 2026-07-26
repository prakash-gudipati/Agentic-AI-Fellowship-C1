"""
Session 36 — agent.py

The Phase 5 agent, now wired as an MCP CLIENT.

This is the join between the two halves of the session. The agent does not own
its tools any more. At startup it asks an MCP server "what can you do?", turns
the answer into its tool list, and then runs the same decide → act → observe
loop you have seen since Session 29 — except every "act" is a tool call sent
over the MCP protocol instead of a local Python function call.

The loop is deliberately the SAME shape as S29/S30. The only new idea is WHERE
the tools come from: discovered from a server, not hard-coded in the agent.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from llm_client import LLMClient
from mcp_client import MCPClient, tools_as_anthropic_schema
from prompts import AGENT_SYSTEM


EventHook = Optional[Callable[[str, Dict], None]]


class MCPAgent:
    """An agent whose entire toolbox is supplied by an MCP server."""

    def __init__(
        self,
        mcp: MCPClient,
        llm: Optional[LLMClient] = None,
        max_steps: int = 6,
        on_event: EventHook = None,
    ) -> None:
        self.mcp = mcp
        self.llm = llm or LLMClient()
        self.max_steps = max_steps  # S31's Max-Step Ceiling still applies
        self.on_event = on_event
        self.tool_calls = 0

        # DISCOVERY: ask the server what it offers, once, at startup.
        self.tool_specs = self.mcp.list_tools()
        self.tools_schema = tools_as_anthropic_schema(self.tool_specs)
        self._emit("discover", {"count": len(self.tool_specs)})

    def answer(self, question: str) -> str:
        """Run the decide → act → observe loop until a final answer."""

        messages: List[Dict] = [{"role": "user", "content": question}]

        for _ in range(self.max_steps):
            decision = self.llm.decide_next(
                system=AGENT_SYSTEM, messages=messages, tools=self.tools_schema
            )

            if decision.kind == "final":
                self._emit("stats", {
                    "tool_calls": self.tool_calls,
                    "llm_calls": self.llm.llm_calls,
                })
                return decision.text

            # decision.kind == "tool_call" — execute it OVER MCP.
            if decision.text:
                self._emit("think", {"text": decision.text})
            self._emit("tool_call", {
                "name": decision.tool_name, "args": decision.tool_args,
            })
            observation = self.mcp.call_tool(decision.tool_name, decision.tool_args)
            self.tool_calls += 1
            self._emit("observation", {"text": observation})

            # Append the assistant tool_use + the user tool_result, exactly as
            # the Anthropic Messages API expects, so the next decide() sees the
            # full history.
            messages.append({
                "role": "assistant",
                "content": decision.raw_assistant_blocks or [
                    {
                        "type": "tool_use",
                        "id": decision.tool_use_id,
                        "name": decision.tool_name,
                        "input": decision.tool_args,
                    }
                ],
            })
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": decision.tool_use_id,
                        "content": observation,
                    }
                ],
            })

        # Hit the ceiling without a final answer.
        self._emit("stats", {
            "tool_calls": self.tool_calls, "llm_calls": self.llm.llm_calls,
        })
        return "Stopped after the maximum number of steps without a final answer."

    def _emit(self, kind: str, payload: Dict) -> None:
        if self.on_event is not None:
            self.on_event(kind, payload)
