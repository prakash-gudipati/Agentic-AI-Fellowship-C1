"""Session 40 — OUTPUT guardrails: validate the model's answer BEFORE the user.

Each class exposes .check(text, context=None) -> GuardrailResult. WHY output-side:
the model is non-deterministic — it can leak PII, drift off-format, say something
unsafe, or hallucinate. The Output Validation Gate is the last line of defense.
Two of these are LLM-as-judge guardrails (Safety, Grounding) routed through
llm_client.complete, so they run offline + deterministically under FAKE_LLM.
"""
from __future__ import annotations

import json
import re

from guardrail_types import Decision, GuardrailResult, Severity
from prompts import GROUNDING_JUDGE, SAFETY_JUDGE
import llm_client


class PIIRedactionGuardrail:
    """Redact any email/card that leaks into the OUTPUT (defense in depth)."""

    EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,16}(?!\d)")

    def check(self, text: str, context=None) -> GuardrailResult:
        transformed = self.EMAIL.sub("[REDACTED_EMAIL]", text)
        transformed = self.CARD.sub("[REDACTED_CARD]", transformed)
        if transformed != text:
            return GuardrailResult(
                name="PIIRedactionGuardrail",
                decision=Decision.REDACT,
                severity=Severity.MEDIUM,
                reason="PII leaked in model output — redacted.",
                transformed_text=transformed,
            )
        return GuardrailResult("PIIRedactionGuardrail", Decision.ALLOW, Severity.LOW, "No PII in output.")


class SchemaGuardrail:
    """BLOCK empty / over-long / system-prompt-leaking output (shape check)."""

    LEAK_FRAGMENTS = ("agent_system", "you are a customer-support assistant", "system prompt")

    def __init__(self, max_chars: int = 1500):
        self.max_chars = max_chars

    def check(self, text: str, context=None) -> GuardrailResult:
        if not text or not text.strip():
            return GuardrailResult("SchemaGuardrail", Decision.BLOCK, Severity.MEDIUM, "Empty output.")
        if len(text) > self.max_chars:
            return GuardrailResult("SchemaGuardrail", Decision.BLOCK, Severity.LOW,
                                   f"Output {len(text)} chars exceeds {self.max_chars}.")
        low = text.lower()
        for frag in self.LEAK_FRAGMENTS:
            if frag in low:
                return GuardrailResult("SchemaGuardrail", Decision.BLOCK, Severity.HIGH,
                                       f"Output leaks a system-prompt fragment: '{frag}'.")
        return GuardrailResult("SchemaGuardrail", Decision.ALLOW, Severity.LOW, "Output shape OK.")


class SafetyGuardrail:
    """LLM-as-judge: BLOCK when the SAFETY_JUDGE returns verdict UNSAFE."""

    def check(self, text: str, context=None) -> GuardrailResult:
        raw = llm_client.complete(SAFETY_JUDGE, text)
        verdict = self._parse(raw)
        if verdict.get("verdict") == "UNSAFE":
            return GuardrailResult(
                name="SafetyGuardrail",
                decision=Decision.BLOCK,
                severity=Severity.HIGH,
                reason="Safety judge: " + str(verdict.get("reason", "unsafe")),
            )
        return GuardrailResult("SafetyGuardrail", Decision.ALLOW, Severity.LOW, "Safety judge: SAFE.")

    @staticmethod
    def _parse(raw: str) -> dict:
        try:
            return json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        except Exception:
            # WHY default UNSAFE: a judge we can't parse is a judge we can't trust.
            return {"verdict": "UNSAFE", "reason": "Unparseable judge response."}


class GroundingGuardrail:
    """LLM-as-judge: BLOCK when the answer is not grounded in `context`.

    This is the hallucination catch — the headline output guardrail. It sends
    CONTEXT + ANSWER to the GROUNDING_JUDGE and blocks on grounded:false.
    """

    def check(self, text: str, context=None) -> GuardrailResult:
        ctx = context or ""
        message = f"CONTEXT: {ctx}\nANSWER: {text}"
        raw = llm_client.complete(GROUNDING_JUDGE, message)
        verdict = self._parse(raw)
        if verdict.get("grounded") is False:
            return GuardrailResult(
                name="GroundingGuardrail",
                decision=Decision.BLOCK,
                severity=Severity.HIGH,
                reason="Grounding judge: " + str(verdict.get("reason", "ungrounded")),
            )
        return GuardrailResult("GroundingGuardrail", Decision.ALLOW, Severity.LOW, "Grounding judge: grounded.")

    @staticmethod
    def _parse(raw: str) -> dict:
        try:
            return json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        except Exception:
            # Fail closed: an unparseable grounding verdict is treated as ungrounded.
            return {"grounded": False, "reason": "Unparseable judge response."}
