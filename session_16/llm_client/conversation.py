"""
Conversation — multi-turn helper. Unchanged from S15.

Holds an ordered list of {role, content} messages and feeds them back
to the same LLMClient on every turn so the model sees the full history.
"""

from __future__ import annotations
from typing import Optional

from .client import LLMClient


class Conversation:
    def __init__(self, client: LLMClient, system: Optional[str] = None) -> None:
        self.client = client
        self.system = system
        self.messages: list[dict] = []

    def say(self, user_message: str, *, max_tokens: int = 1024) -> str:
        self.messages.append({"role": "user", "content": user_message})
        # Note: complete_messages on the LLMClient is the multi-turn path.
        # If your LLMClient is the S15 version, copy that method over.
        if not hasattr(self.client, "complete_messages"):
            raise AttributeError(
                "LLMClient.complete_messages is required for Conversation. "
                "Carry it over from your S15 wrapper."
            )
        reply = self.client.complete_messages(
            self.messages, system=self.system, max_tokens=max_tokens
        )
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def transcript(self) -> str:
        lines = []
        if self.system:
            lines.append(f"[SYSTEM] {self.system}")
        for m in self.messages:
            lines.append(f"[{m['role'].upper()}] {m['content']}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.messages = []

    def turn_count(self) -> int:
        return len(self.messages) // 2
