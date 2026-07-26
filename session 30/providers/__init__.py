"""
Session 30 — providers/

Two provider adapters, one interface. Same agent loop runs on either.

  providers/
    __init__.py            ← convenience re-exports
    base.py                ← abstract Provider interface + shared dataclasses
    anthropic_provider.py  ← Anthropic Messages API (tool_use blocks)
    openai_provider.py     ← OpenAI Chat Completions API (tool_calls)

Why split into a package?
  - Provider SDKs change. Their request/response shapes change. The agent
    loop in agent.py doesn't care which one we use — it asks for the next
    step and gets back a clean dataclass either way. The adapter is where
    every provider-specific quirk goes to die.
"""

from providers.base import (
    AssistantMessage,
    AssistantToolCall,
    Provider,
    ToolResultMessage,
)
from providers.anthropic_provider import AnthropicProvider
from providers.openai_provider import OpenAIProvider


__all__ = [
    "AssistantMessage",
    "AssistantToolCall",
    "Provider",
    "ToolResultMessage",
    "AnthropicProvider",
    "OpenAIProvider",
]
