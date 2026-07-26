"""
Session 35 — messages.py

PROD PATTERN: Inter-Agent Message Schema.

This is the typed message bus that carries every cross-agent exchange
in S35. It is deliberately tiny — under 150 lines — because the value
is in the SHAPE, not the machinery.

WHY this exists:
  - In S34 we had a Shared Scratchpad (typed sections — FACTS, DRAFT,
    CRITIQUE, FINAL_ANSWER). Good for a small flat crew.
  - In S35 we have a hierarchy (Director → Managers → Workers) AND
    we have debates (multiple peers exchanging arguments). A flat
    scratchpad does not capture WHO is talking to WHOM.
  - The Message bus replaces the scratchpad as the central state when
    the agent graph is no longer a star.

DESIGN PROPERTIES:
  - Append-only. Once a Message is in the bus, it stays. Replay is
    free.
  - Every Message names a sender + recipient. Routing is explicit.
  - Every Message names an intent. The intent enumeration is small
    on purpose (ASSIGN / REPORT / REVIEW / ESCALATE / ACK). If you
    feel the need to add a sixth intent, redesign first.
  - inbox(recipient) returns only messages addressed to that recipient.
    Agents read their own mail; they do NOT read the whole bus.
"""

from __future__ import annotations

import uuid
from typing import Iterable, List, Optional

from agent_types import Message


class MessageBus:
    """Append-only typed message bus.

    Usage pattern:
        bus = MessageBus()
        bus.send(Message(sender=..., recipient=..., intent="ASSIGN", ...))
        for m in bus.inbox("researcher"):
            ...
    """

    def __init__(self) -> None:
        self._messages: List[Message] = []
        self._read_cursors: dict = {}     # recipient -> int (next to read)

    # ------------------------------------------------------------------
    # Send / receive
    # ------------------------------------------------------------------

    def send(self, msg: Message) -> Message:
        """Append a message to the bus. Assigns an id if missing."""

        if not msg.msg_id:
            msg.msg_id = f"m_{len(self._messages):04d}_{uuid.uuid4().hex[:6]}"
        self._messages.append(msg)
        return msg

    def inbox(self, recipient: str) -> List[Message]:
        """Return all messages ever addressed to this recipient."""

        return [m for m in self._messages if m.recipient == recipient]

    def unread(self, recipient: str) -> List[Message]:
        """Return only messages this recipient has not yet read.

        Recipients call mark_read() after they process an unread batch.
        """

        start = self._read_cursors.get(recipient, 0)
        unread: List[Message] = []
        for i, m in enumerate(self._messages):
            if i < start:
                continue
            if m.recipient == recipient:
                unread.append(m)
        return unread

    def mark_read(self, recipient: str) -> None:
        """Advance the read cursor for this recipient to end-of-bus."""

        self._read_cursors[recipient] = len(self._messages)

    # ------------------------------------------------------------------
    # Inspection / replay
    # ------------------------------------------------------------------

    def all_messages(self) -> List[Message]:
        return list(self._messages)

    def by_sender(self, sender: str) -> List[Message]:
        return [m for m in self._messages if m.sender == sender]

    def by_intent(self, intent: str) -> List[Message]:
        return [m for m in self._messages if m.intent == intent]

    def __len__(self) -> int:
        return len(self._messages)

    # ------------------------------------------------------------------
    # Convenience constructors — these enforce the schema at call sites
    # so individual agents don't have to remember the field order.
    # ------------------------------------------------------------------

    def assign(self, sender: str, recipient: str, subject: str,
               payload: Optional[dict] = None, round_num: int = 0) -> Message:
        return self.send(Message(
            sender=sender, recipient=recipient, intent="ASSIGN",
            subject=subject, payload=payload or {}, round_num=round_num,
        ))

    def report(self, sender: str, recipient: str, subject: str,
               payload: Optional[dict] = None, round_num: int = 0) -> Message:
        return self.send(Message(
            sender=sender, recipient=recipient, intent="REPORT",
            subject=subject, payload=payload or {}, round_num=round_num,
        ))

    def review(self, sender: str, recipient: str, subject: str,
               payload: Optional[dict] = None, round_num: int = 0) -> Message:
        return self.send(Message(
            sender=sender, recipient=recipient, intent="REVIEW",
            subject=subject, payload=payload or {}, round_num=round_num,
        ))

    def escalate(self, sender: str, recipient: str, subject: str,
                 payload: Optional[dict] = None, round_num: int = 0) -> Message:
        return self.send(Message(
            sender=sender, recipient=recipient, intent="ESCALATE",
            subject=subject, payload=payload or {}, round_num=round_num,
        ))

    def ack(self, sender: str, recipient: str, subject: str,
            payload: Optional[dict] = None, round_num: int = 0) -> Message:
        return self.send(Message(
            sender=sender, recipient=recipient, intent="ACK",
            subject=subject, payload=payload or {}, round_num=round_num,
        ))

    # ------------------------------------------------------------------
    # Validation helpers — used by debug demos
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of human-readable schema violations.

        Empty list means clean. Used in the wire-format debug demo.
        """

        problems: List[str] = []
        for m in self._messages:
            if m.intent not in Message.ALLOWED_INTENTS:
                problems.append(
                    f"{m.msg_id}: bad intent '{m.intent}'"
                )
            if not m.subject:
                problems.append(
                    f"{m.msg_id}: empty subject (sender={m.sender})"
                )
        return problems
