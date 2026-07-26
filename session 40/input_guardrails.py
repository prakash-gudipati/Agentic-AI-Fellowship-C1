"""Session 40 — INPUT guardrails: runtime, per-request defense BEFORE the model.

Each class exposes .check(text) -> GuardrailResult. They are deliberately small
and single-purpose so the chain can compose them in any order. WHY input-side:
the cheapest place to stop an attack is before you spend a model call on it.
"""
from __future__ import annotations

import re

from guardrail_types import Decision, GuardrailResult, Severity


class InjectionGuardrail:
    """Catch prompt-injection phrases that try to override the system prompt."""

    PHRASES = (
        "ignore previous instructions",
        "ignore all previous",
        "disregard",
        "system prompt",
        "reveal your instructions",
        "reveal your system",
    )

    def check(self, text: str) -> GuardrailResult:
        low = text.lower()
        for phrase in self.PHRASES:
            if phrase in low:
                return GuardrailResult(
                    name="InjectionGuardrail",
                    decision=Decision.BLOCK,
                    severity=Severity.HIGH,
                    reason=f"Prompt-injection phrase detected: '{phrase}'.",
                )
        return GuardrailResult("InjectionGuardrail", Decision.ALLOW, Severity.LOW, "No injection phrase.")


class PIIGuardrail:
    """Redact emails/phones; BLOCK on a full card-like number (never store it)."""

    EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s-]?){10,13}(?!\d)")
    CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,16}(?!\d)")

    def check(self, text: str) -> GuardrailResult:
        # A full card number is CRITICAL — we must not let it reach the model.
        if self.CARD.search(re.sub(r"[ -]", "", text)) or self._card_in(text):
            return GuardrailResult(
                name="PIIGuardrail",
                decision=Decision.BLOCK,
                severity=Severity.CRITICAL,
                reason="Full card-like number detected — blocked, never processed.",
            )
        transformed = self.EMAIL.sub("[REDACTED_EMAIL]", text)
        transformed = self.PHONE.sub("[REDACTED_PHONE]", transformed)
        if transformed != text:
            return GuardrailResult(
                name="PIIGuardrail",
                decision=Decision.REDACT,
                severity=Severity.MEDIUM,
                reason="Email/phone redacted before processing.",
                transformed_text=transformed,
            )
        return GuardrailResult("PIIGuardrail", Decision.ALLOW, Severity.LOW, "No PII found.")

    def _card_in(self, text: str) -> bool:
        """True if stripping separators leaves a 13-16 digit run (a card)."""
        digits = re.sub(r"\D", "", text)
        return bool(re.search(r"\d{13,16}", digits))


class LengthGuardrail:
    """BLOCK over-long input — a cheap denial-of-wallet / abuse brake."""

    def __init__(self, max_chars: int = 2000):
        self.max_chars = max_chars

    def check(self, text: str) -> GuardrailResult:
        if len(text) > self.max_chars:
            return GuardrailResult(
                name="LengthGuardrail",
                decision=Decision.BLOCK,
                severity=Severity.LOW,
                reason=f"Input {len(text)} chars exceeds limit {self.max_chars}.",
            )
        return GuardrailResult("LengthGuardrail", Decision.ALLOW, Severity.LOW, "Within length limit.")


class TopicGuardrail:
    """BLOCK off-scope topics — a payments bot must not give medical/legal advice."""

    DENYLIST = ("medical", "legal advice", "investment tip", "diagnos", "prescri")

    def check(self, text: str) -> GuardrailResult:
        low = text.lower()
        for term in self.DENYLIST:
            if term in low:
                return GuardrailResult(
                    name="TopicGuardrail",
                    decision=Decision.BLOCK,
                    severity=Severity.MEDIUM,
                    reason=f"Off-scope topic for a payments bot: '{term}'.",
                )
        return GuardrailResult("TopicGuardrail", Decision.ALLOW, Severity.LOW, "On-topic.")
