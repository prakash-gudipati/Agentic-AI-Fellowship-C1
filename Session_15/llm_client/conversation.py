"""
Conversation — a multi-turn helper on top of LLMClient.

Most LLM apps eventually need conversation history: chat assistants,
agents (Phase 5), follow-up questions on retrieved documents (Phase 4
RAG). This helper hides the bookkeeping.

Usage:

    chat = Conversation(client, system="You are a helpful tutor.")
    chat.say("What is a vending machine?")          # turn 1
    chat.say("How is it different from a fridge?")  # turn 2 — has memory
    print(chat.transcript())

Why a separate class instead of a method on LLMClient? Single Responsibility.
LLMClient handles the network call. Conversation handles the message
history. Mixing them would make both harder to test.
"""

from __future__ import annotations
from typing import Optional

from .client import LLMClient


class Conversation:
    """Holds an ordered list of {role, content} messages and feeds them
    back to the same LLMClient on every turn so the model sees the full
    history."""

    def __init__(self, client: LLMClient, system: Optional[str] = None) -> None:
        self.client = client
        self.system = system
        self.messages: list[dict] = []

    def say(self, user_message: str, *, max_tokens: int = 1024) -> str:
        """Send the next user turn. Returns the model's reply.

        Side effect: the user message AND the model reply are appended
        to self.messages, so the next say() sees both.
        """
        self.messages.append({"role": "user", "content": user_message})
        reply = self.client.complete_messages(
            self.messages, system=self.system, max_tokens=max_tokens
        )
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def transcript(self) -> str:
        """Pretty-print the conversation so far. Useful in tests and logs."""
        lines = []
        if self.system:
            lines.append(f"[SYSTEM] {self.system}")
        for m in self.messages:
            lines.append(f"[{m['role'].upper()}] {m['content']}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Forget all turns. Keeps the same client and system prompt."""
        self.messages = []

    def turn_count(self) -> int:
        """Number of completed user/assistant turn pairs."""
        return len(self.messages) // 2
