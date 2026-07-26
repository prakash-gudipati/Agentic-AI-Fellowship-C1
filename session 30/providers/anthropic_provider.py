"""
Session 30 - providers/anthropic_provider.py

Anthropic Messages API adapter. Wraps the native tool_use mechanism.

  Our Tool                                 Anthropic schema
  --------                                 -----------------
  name                                     "name"
  description                              "description"
  input_schema (JSON Schema)               "input_schema"

  Response normalisation:
  response.content = [block, block]   ->  walk each block:
      block.type == "text"        ->  AssistantMessage.text
      block.type == "tool_use"    ->  AssistantMessage.tool_calls[]

  Sending results back:
  Anthropic expects a user-role message whose content is a list of
  tool_result blocks, one per call, each carrying the original tool_use_id.

  tool_choice (S30 expansion):
  None / "auto"          -> omit param (default)
  "any"                  -> {"type": "any"}
  {"name": "<tool>"}     -> {"type": "tool", "name": "<tool>"}
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from anthropic import Anthropic

from providers.base import (
    AssistantMessage,
    AssistantToolCall,
    Provider,
    ToolResultMessage,
)


DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider(Provider):
    """Anthropic Messages adapter."""

    name = "anthropic"

    def __init__(self, *, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def format_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Anthropic expects {name, description, input_schema} per tool."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]

    def next_turn(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools_formatted: List[Dict[str, Any]],
        tool_choice: Any = None,
    ) -> AssistantMessage:
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            tools=tools_formatted,
            messages=messages,
        )
        if tool_choice is None or tool_choice == "auto":
            pass
        elif tool_choice == "any":
            kwargs["tool_choice"] = {"type": "any"}
        elif isinstance(tool_choice, dict) and "name" in tool_choice:
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice["name"]}
        else:
            raise ValueError(f"Unsupported tool_choice for Anthropic: {tool_choice!r}")

        response = self.client.messages.create(**kwargs)

        text_parts: List[str] = []
        tool_calls: List[AssistantToolCall] = []

        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                tool_calls.append(
                    AssistantToolCall(
                        call_id=block.id,
                        tool_name=block.name,
                        tool_args=dict(block.input) if block.input else {},
                    )
                )

        return AssistantMessage(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            raw=response,
        )

    def build_tool_result_message(self, tool_results: ToolResultMessage) -> Dict[str, Any]:
        """Anthropic: one user-role message whose content is a list of tool_result blocks."""
        content_blocks: List[Dict[str, Any]] = []
        for result in tool_results.results:
            content_blocks.append({
                "type": "tool_result",
                "tool_use_id": result["call_id"],
                "content": result["content"],
            })
        return {"role": "user", "content": content_blocks}

    @staticmethod
    def assistant_message_from_response(message: AssistantMessage) -> Dict[str, Any]:
        """Anthropic requires the assistant's turn to be sent back verbatim."""
        blocks: List[Dict[str, Any]] = []
        if message.text:
            blocks.append({"type": "text", "text": message.text})
        for call in message.tool_calls:
            blocks.append({
                "type": "tool_use",
                "id": call.call_id,
                "name": call.tool_name,
                "input": call.tool_args,
            })
        return {"role": "assistant", "content": blocks}
