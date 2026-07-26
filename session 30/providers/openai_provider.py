"""
Session 30 - providers/openai_provider.py

OpenAI Chat Completions adapter. Wraps the native tool_calls mechanism.

  Our Tool                                 OpenAI tool schema
  --------                                 -------------------
  name                                     "function.name"
  description                              "function.description"
  input_schema (JSON Schema)               "function.parameters"
  (wrapped inside {"type": "function", "function": {...}})

  Response normalisation:
  response.choices[0].message.tool_calls   ->  list of ToolCall objects
      .id                                  ->  call_id
      .function.name                       ->  tool_name
      .function.arguments  (JSON STRING)   ->  parsed -> tool_args dict

  Sending results back:
  OpenAI expects ONE role="tool" message PER tool result, each carrying
  the original tool_call_id.

  tool_choice (S30 expansion):
  None / "auto"          -> omit param (default)
  "any"                  -> "required"
  {"name": "<tool>"}     -> {"type": "function", "function": {"name": "<tool>"}}
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from providers.base import (
    AssistantMessage,
    AssistantToolCall,
    Provider,
    ToolResultMessage,
)


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 1024


class OpenAIProvider(Provider):
    """OpenAI Chat Completions adapter with tool_calls."""

    name = "openai"

    def __init__(self, *, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def format_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """OpenAI expects tools wrapped in {type:function, function:{...}}."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
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
        wire_messages = [{"role": "system", "content": system_prompt}] + messages

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=wire_messages,
            tools=tools_formatted,
        )
        if tool_choice is None or tool_choice == "auto":
            pass
        elif tool_choice == "any":
            kwargs["tool_choice"] = "required"
        elif isinstance(tool_choice, dict) and "name" in tool_choice:
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
        else:
            raise ValueError(f"Unsupported tool_choice for OpenAI: {tool_choice!r}")

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message

        tool_calls: List[AssistantToolCall] = []
        for tc in (message.tool_calls or []):
            try:
                tool_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {"_raw_arguments": tc.function.arguments}
            tool_calls.append(
                AssistantToolCall(
                    call_id=tc.id,
                    tool_name=tc.function.name,
                    tool_args=tool_args,
                )
            )

        stop_reason_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }
        stop_reason = stop_reason_map.get(
            choice.finish_reason or "stop", choice.finish_reason or "end_turn"
        )

        return AssistantMessage(
            text=(message.content or "").strip(),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw=response,
        )

    def build_tool_result_message(self, tool_results: ToolResultMessage) -> Dict[str, Any]:
        """OpenAI: ONE role=tool message PER result.

        The agent loop knows it gets a list back from OpenAI but a single dict
        from Anthropic. The "_messages" wrapper key signals "extend, don't append."
        """
        out: List[Dict[str, Any]] = []
        for result in tool_results.results:
            out.append({
                "role": "tool",
                "tool_call_id": result["call_id"],
                "content": result["content"],
            })
        return {"_messages": out}

    @staticmethod
    def assistant_message_from_response(message: AssistantMessage) -> Dict[str, Any]:
        """OpenAI expects the assistant turn echoed back with content + tool_calls."""
        tool_calls_payload = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.tool_name,
                    "arguments": json.dumps(call.tool_args),
                },
            }
            for call in message.tool_calls
        ]
        return {
            "role": "assistant",
            "content": message.text or None,
            "tool_calls": tool_calls_payload or None,
        }
