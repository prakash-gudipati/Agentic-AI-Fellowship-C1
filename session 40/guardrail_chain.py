"""Session 40 — the GuardrailChain: compose checks, short-circuit, fail closed.

PROD PATTERN: Input Guardrail Chain / Output Validation Gate — run guardrails in
sequence; the FIRST BLOCK short-circuits the whole request. REDACT updates the
working text and continues (so later checks see the cleaned text).

PROD PATTERN: Fail-Closed Default — if a guardrail RAISES an exception, we do NOT
silently pass the request through. With fail_closed=True (the production default)
an error becomes a synthetic BLOCK(CRITICAL). A safety layer that fails OPEN is
not a safety layer.
"""
from __future__ import annotations

from typing import List, Optional

from guardrail_types import Decision, GuardrailReport, GuardrailResult, Severity


class GuardrailChain:
    """Runs an ordered list of guardrails and aggregates one GuardrailReport."""

    def __init__(self, guardrails: List, stage: str = "input", fail_closed: bool = True):
        self.guardrails = guardrails
        self.stage = stage
        self.fail_closed = fail_closed

    def run(self, text: str, context: Optional[str] = None) -> GuardrailReport:
        """Execute the chain. First BLOCK wins; REDACT mutates the working text."""
        report = GuardrailReport(stage=self.stage)
        working = text

        for guard in self.guardrails:
            try:
                result = self._invoke(guard, working, context)
            except Exception as exc:  # a guardrail itself broke
                result = self._on_error(guard, exc)
                report.results.append(result)
                if result.decision == Decision.BLOCK:
                    # Fail-closed path: stop the request right here.
                    report.final_decision = Decision.BLOCK
                    report.final_text = working
                    return report
                # fail_closed=False: log via the result, then continue.
                continue

            report.results.append(result)

            if result.decision == Decision.BLOCK:
                # First BLOCK short-circuits — no point running later checks.
                report.final_decision = Decision.BLOCK
                report.final_text = working
                return report

            if result.decision == Decision.REDACT and result.transformed_text is not None:
                # Carry the cleaned text forward so the next guardrail sees it.
                working = result.transformed_text

        report.final_decision = Decision.ALLOW
        report.final_text = working
        return report

    @staticmethod
    def _invoke(guard, text: str, context: Optional[str]) -> GuardrailResult:
        """Input guards take .check(text); output guards take .check(text, context)."""
        try:
            return guard.check(text, context)
        except TypeError:
            return guard.check(text)

    def _on_error(self, guard, exc: Exception) -> GuardrailResult:
        """Turn a raised exception into a decision per the fail-closed policy."""
        name = getattr(guard, "__class__", type(guard)).__name__
        if self.fail_closed:
            return GuardrailResult(
                name=name,
                decision=Decision.BLOCK,
                severity=Severity.CRITICAL,
                reason=f"Guardrail error (fail-closed → BLOCK): {exc}",
            )
        # fail_closed=False — record the error but let the request pass.
        return GuardrailResult(
            name=name,
            decision=Decision.ALLOW,
            severity=Severity.HIGH,
            reason=f"Guardrail error (fail-open → ALLOW): {exc}",
        )
