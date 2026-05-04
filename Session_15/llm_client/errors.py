"""
Custom exception hierarchy for LLMClient.

Why a hierarchy? Because real production code needs to react differently
to different kinds of failure. A 401 (bad key) needs a different response
than a 429 (rate limit) than a 500 (server error). Naming each one and
making them all inherit from LLMError lets the caller catch broadly when
they want, and specifically when they need to.

Inheritance order:
    LLMError                              ← base; catch this for "anything from the wrapper"
      ├─ ProviderTimeout                  ← retryable: try again
      ├─ ProviderUnavailable              ← retryable: try again or fall back
      ├─ RateLimitHit (retry_after)       ← retryable: wait then try again
      ├─ InvalidAPIKey                    ← NOT retryable: fail fast
      ├─ ContextLengthExceeded            ← NOT retryable: prompt is too long
      ├─ ContentFilterTriggered           ← NOT retryable: rewrite the prompt
      ├─ BudgetExceeded                   ← NOT retryable: caller pre-set a cost cap
      └─ AllProvidersFailed               ← raised when fallback chain exhausts
"""

from __future__ import annotations
from typing import Optional


class LLMError(Exception):
    """Base class for every error LLMClient raises.

    Catching this catches everything the wrapper produces. Useful for
    application-level error reporting where you do not need to react
    to specific failure modes.
    """


# ─── retryable ─────────────────────────────────────────────────────────────


class ProviderTimeout(LLMError):
    """The provider did not respond within our timeout window. Retry."""


class ProviderUnavailable(LLMError):
    """5xx response from the provider. Provider has a problem. Retry, and
    consider falling over to the secondary provider if one is configured.
    """


class RateLimitHit(LLMError):
    """The provider is throttling us (HTTP 429). Wait, then retry.

    Where possible we surface the provider's recommended `retry_after`
    seconds so the caller (or our retry decorator) can honour it.
    """

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# ─── NOT retryable ─────────────────────────────────────────────────────────


class InvalidAPIKey(LLMError):
    """HTTP 401/403. The key is wrong, expired, or lacks permissions.

    Retrying is pointless and looks like an attack. Fail fast so the
    operator can fix the key.
    """


class ContextLengthExceeded(LLMError):
    """The prompt + expected output exceeds the model's context window.

    The caller has to shorten the prompt or switch models. The wrapper
    cannot fix this on its own.
    """


class ContentFilterTriggered(LLMError):
    """The provider's safety filter blocked the request or response.

    Caller decides how to react: rewrite, log to a moderation queue, or
    surface a generic message to the end user.
    """


class BudgetExceeded(LLMError):
    """The wrapper hit a caller-configured cost cap.

    Used as a guardrail in long-running scripts and in tenant-bounded
    SaaS environments. Caller decides whether to keep going on a different
    budget or stop.
    """


class AllProvidersFailed(LLMError):
    """Every configured provider in the fallback chain raised a permanent
    error. The wrapper has no more options. Caller decides what to show
    the end user.
    """
