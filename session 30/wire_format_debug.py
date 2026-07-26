"""
Session 30 - wire_format_debug.py

The provider_comparison slide shows a schematic side-by-side of Anthropic
vs OpenAI. This module shows the REAL JSON bytes - exactly what each SDK
puts on the network when you make a tool-calling request.

We don't actually fire the network call. We construct the same payload
the SDK would send, print it as pretty JSON, and stop. Zero API cost.
Run it once and you'll never have to wonder again what tool-calling
"really" looks like on the wire.

Run:
    python wire_format_debug.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from providers.anthropic_provider import AnthropicProvider
from providers.openai_provider import OpenAIProvider
from tools import SHARP_TOOLS


# A single example conversation. ONE user question, the assistant about to
# call a tool. This is the request the SDK would send for the FIRST turn.
EXAMPLE_USER_QUESTION = "What is the GDP of India in 2024?"


def anthropic_request_payload() -> Dict[str, Any]:
    """Construct the EXACT dict Anthropic's SDK would serialize and POST."""
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "DEBUG_KEY_NEVER_SENT")
    provider = AnthropicProvider()
    formatted_tools = provider.format_tools(SHARP_TOOLS)
    return {
        "url": "https://api.anthropic.com/v1/messages",
        "method": "POST",
        "headers": {
            "x-api-key": "<your ANTHROPIC_API_KEY>",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        "body": {
            "model": provider.model,
            "max_tokens": provider.max_tokens,
            "system": "You are a careful research agent...",
            "tools": formatted_tools,
            "messages": [
                {"role": "user", "content": EXAMPLE_USER_QUESTION},
            ],
            # tool_choice is OMITTED on auto mode (default).
        },
    }


def anthropic_response_payload() -> Dict[str, Any]:
    """What Anthropic returns when the model wants to call web_search."""
    return {
        "id": "msg_01ABC123",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5-20251001",
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "text",
                "text": "I'll look up India's 2024 GDP.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01XYZ789",
                "name": "web_search",
                "input": {"query": "GDP of India 2024"},
            },
        ],
        "usage": {"input_tokens": 423, "output_tokens": 47},
    }


def anthropic_next_user_message() -> Dict[str, Any]:
    """The follow-up message we POST back, carrying the tool result."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01XYZ789",
                "content": "World Bank: India's nominal GDP for 2024 is approximately USD 3,937,011,000,000.",
            }
        ],
    }


def openai_request_payload() -> Dict[str, Any]:
    """Construct the EXACT dict OpenAI's SDK would serialize and POST."""
    import os
    os.environ.setdefault("OPENAI_API_KEY", "DEBUG_KEY_NEVER_SENT")
    provider = OpenAIProvider()
    formatted_tools = provider.format_tools(SHARP_TOOLS)
    return {
        "url": "https://api.openai.com/v1/chat/completions",
        "method": "POST",
        "headers": {
            "Authorization": "Bearer <your OPENAI_API_KEY>",
            "content-type": "application/json",
        },
        "body": {
            "model": provider.model,
            "max_tokens": provider.max_tokens,
            "messages": [
                {"role": "system", "content": "You are a careful research agent..."},
                {"role": "user", "content": EXAMPLE_USER_QUESTION},
            ],
            "tools": formatted_tools,
        },
    }


def openai_response_payload() -> Dict[str, Any]:
    """What OpenAI returns when the model wants to call web_search."""
    return {
        "id": "chatcmpl-AbC123",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_xyz789",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                # IMPORTANT: arguments is a JSON STRING, not an object.
                                "arguments": "{\"query\": \"GDP of India 2024\"}",
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 401, "completion_tokens": 21, "total_tokens": 422},
    }


def openai_next_messages() -> List[Dict[str, Any]]:
    """OpenAI: one role='tool' message PER result. Compare with Anthropic above."""
    return [
        {
            "role": "tool",
            "tool_call_id": "call_xyz789",
            "content": "World Bank: India's nominal GDP for 2024 is approximately USD 3,937,011,000,000.",
        }
    ]


def main() -> None:
    print("=" * 80)
    print("ANTHROPIC - REQUEST (turn 1)")
    print("=" * 80)
    print(json.dumps(anthropic_request_payload(), indent=2))

    print("\n" + "=" * 80)
    print("ANTHROPIC - RESPONSE (turn 1)")
    print("=" * 80)
    print(json.dumps(anthropic_response_payload(), indent=2))

    print("\n" + "=" * 80)
    print("ANTHROPIC - NEXT USER MESSAGE (turn 2: carrying tool result)")
    print("=" * 80)
    print(json.dumps(anthropic_next_user_message(), indent=2))

    print("\n\n" + "#" * 80)
    print("# DIFFERENCES TO NOTICE")
    print("#" * 80)
    print("# 1. Tool schema shape:")
    print("#    Anthropic: {name, description, input_schema}")
    print("#    OpenAI:    {type:'function', function:{name, description, parameters}}")
    print("# 2. tool_calls live INSIDE response.content blocks (Anthropic) vs")
    print("#    in a separate .tool_calls field (OpenAI).")
    print("# 3. OpenAI's arguments is a JSON STRING - parse it before use.")
    print("# 4. Tool RESULT message:")
    print("#    Anthropic: ONE user message with a list of tool_result blocks.")
    print("#    OpenAI:    ONE role='tool' message PER result.")
    print("#" * 80)

    print("\n" + "=" * 80)
    print("OPENAI - REQUEST (turn 1)")
    print("=" * 80)
    print(json.dumps(openai_request_payload(), indent=2))

    print("\n" + "=" * 80)
    print("OPENAI - RESPONSE (turn 1)")
    print("=" * 80)
    print(json.dumps(openai_response_payload(), indent=2))

    print("\n" + "=" * 80)
    print("OPENAI - NEXT MESSAGES (turn 2: carrying tool result)")
    print("=" * 80)
    print(json.dumps(openai_next_messages(), indent=2))


if __name__ == "__main__":
    main()
