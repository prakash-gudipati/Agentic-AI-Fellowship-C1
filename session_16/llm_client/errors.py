"""
Custom exception hierarchy — Session 16 extension of S15.

S16 adds three new exception types:

    PromptInjectionDetected  ← fail-fast: a defended prompt or output
                                 contained a known injection pattern.
                                 Caller decides what to do (log, escalate,
                                 show generic error, retry with stricter
                                 sanitisation).

    StructuredOutputFailed   ← fail-fast: even with native structured
                                 outputs, the response did not validate
                                 against the supplied schema. Rare with
                                 strict mode, but possible. Caller fixes
                                 the schema or the prompt.

    ReasoningTimeout         ← retryable-but-cautious: reasoning models
                                 (o1, extended thinking) take longer.
                                 We DO retry once, but with a longer
                                 backoff than normal calls.
"""

from __future__ import annotations
from typing import Optional


# ─── S15 base + retryable ─────────────────────────────────────────────────


class LLMError(Exception):
    """Base class for every error LLMClient raises."""


class ProviderTimeout(LLMError):
    """The provider did not respond within our timeout window. Retry."""


class ProviderUnavailable(LLMError):
    """5xx response from the provider. Retry, possibly fall over."""


class RateLimitHit(LLMError):
    """HTTP 429. Wait, then retry. Honour `retry_after` when supplied."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# ─── S16 NEW: reasoning-model timeout ────────────────────────────────────


class ReasoningTimeout(LLMError):
    """A reasoning-model call exceeded its (longer) timeout window.

    Reasoning models like o1 and Claude extended-thinking can take 30+
    seconds for hard problems. We treat this as retryable but bias
    toward a longer wait between attempts.
    """


# ─── S15 fail-fast ─────────────────────────────────────────────────────────


class InvalidAPIKey(LLMError):
    """HTTP 401/403. Operator must fix the key. Do NOT retry."""


class ContextLengthExceeded(LLMError):
    """Prompt + expected output exceeds the model's context window."""


class ContentFilterTriggered(LLMError):
    """Provider's safety filter blocked the request or response."""


class BudgetExceeded(LLMError):
    """Caller-configured cost cap was hit before this call ran."""


class AllProvidersFailed(LLMError):
    """Every configured provider in the fallback chain raised permanently."""


# ─── S16 NEW: structured output failures ─────────────────────────────────


class StructuredOutputFailed(LLMError):
    """The model returned a response that did not validate against the
    Pydantic schema, even though we requested strict structured output.

    Usually means: schema is too restrictive, OR the model is not one
    that supports strict mode. Caller fixes the schema or switches model.
    """


# ─── S16 NEW: prompt-injection defense ───────────────────────────────────


class PromptInjectionDetected(LLMError):
    """A known prompt-injection pattern was detected in either the input
    document or the model output.

    Raised by:
      - sanitize_input() when the user's own prompt contains red-flag patterns
      - output_filter() when the model's response leaks system-prompt or
        appears to comply with an injected instruction

    Caller decides: log to a moderation queue, escalate, show generic
    error, or retry with stricter spotlighting.
    """

    def __init__(self, message: str, *, where: str = "unknown",
                 pattern: Optional[str] = None) -> None:
        super().__init__(message)
        self.where = where        # "input" | "document" | "output"
        self.pattern = pattern    # which regex matched
