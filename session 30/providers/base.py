"""
Session 30 - providers/base.py

The abstract Provider interface. The agent loop talks to THIS, never to
the SDKs directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass
class AssistantToolCall:
    """One tool the model wants us to run on its behalf."""
    call_id: str
    tool_name: str
    tool_args: Dict[str, Any]


@dataclass
class AssistantMessage:
    """One model turn, normalised across providers."""
    text: str
    tool_calls: List[AssistantToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw: Any = None


@dataclass
class ToolResultMessage:
    """The bridge back to the provider: one or more tool results in one message."""
    results: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, *, call_id: str, content: str) -> None:
        self.results.append({"call_id": call_id, "content": content})


class Provider(Protocol):
    """All adapters implement these three methods, nothing more.

    Three things the adapter must do for any provider:
      1. Convert our Tool dataclass into the provider-specific schema format.
      2. Make one model turn and return a normalised AssistantMessage.
      3. Format the tool result for the next request in whatever shape the
         provider's API expects.
    """

    name: str

    def format_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert our Tool list into the schema shape this provider expects."""

    def next_turn(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools_formatted: List[Dict[str, Any]],
        tool_choice: Any = None,
    ) -> AssistantMessage:
        """Make one model call. Return a normalised AssistantMessage.

        tool_choice controls the API's tool-selection mode:
            None or "auto"           - model decides whether to call a tool
            "any"                    - model must call SOME tool
            {"name": "<tool_name>"}  - model must call THIS specific tool
        """

    def build_tool_result_message(
        self,
        tool_results: ToolResultMessage,
    ) -> Dict[str, Any]:
        """Format the tool result(s) as the next user-role message for the provider."""
