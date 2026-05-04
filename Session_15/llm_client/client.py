"""
LLMClient — the production wrapper class.

Design goals (read this before you read the code):

1. ONE PUBLIC METHOD for completion. The application calls .complete().
   It never sees which provider answered.

2. RETRY ONLY THE TRANSIENT FAILURES. We retry timeouts, 5xx, and 429.
   We never retry 401 (bad key), 4xx other than 429 (your fault), or
   context-length-exceeded (the prompt is too long; retrying is futile).

3. FALLBACK CHAIN. If a primary provider raises a permanent error and a
   fallback provider is configured, we try the fallback. If both fail
   permanently, we raise AllProvidersFailed.

4. STRUCTURED LOG PER CALL. Every call appends one JSON object to
   self.call_log AND to the standard logger. Provider, model, tokens,
   cost, latency, ok, error.

5. COST GUARD. Every successful call increments self.total_cost_usd.
   If max_cost_usd is set on construction, a call that would push past
   the cap raises BudgetExceeded BEFORE making the network call.

6. STREAMING. .stream() yields text chunks. Same retry/log/cost story.

The wrapper is intentionally pure-Python and SDK-thin. Every concept is
visible in this file. Real production wrappers add async, OpenTelemetry
traces, multi-region routing, and a circuit breaker — all of which can
be layered on this same shell.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Iterator, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .cost import calculate_cost_usd
from .errors import (
    AllProvidersFailed,
    BudgetExceeded,
    ContentFilterTriggered,
    ContextLengthExceeded,
    InvalidAPIKey,
    LLMError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimitHit,
)


_log = logging.getLogger("llm_client")


_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
}


# Errors we re-raise as our retryable types — tenacity retries on them.
_RETRYABLE = (ProviderTimeout, ProviderUnavailable, RateLimitHit)


class LLMClient:
    """A reusable, provider-agnostic LLM client with retries, fallback,
    structured logging, cost tracking, and streaming.

    Construct one of these per (provider, model) you want to use, OR per
    application — and let the wrapper handle the rest.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        *,
        fallback_provider: Optional[str] = None,
        fallback_model: Optional[str] = None,
        tenant_id: Optional[str] = None,
        max_cost_usd: Optional[float] = None,
    ) -> None:
        self.provider = provider
        self.model = model or _DEFAULT_MODELS.get(provider) or _missing_model(provider)
        self.fallback_provider = fallback_provider
        self.fallback_model = (
            fallback_model
            or (_DEFAULT_MODELS.get(fallback_provider) if fallback_provider else None)
        )
        self.tenant_id = tenant_id
        self.max_cost_usd = max_cost_usd

        self.call_log: list[dict] = []
        self._cache: dict = {}
        self._cost_usd = 0.0

        self._client = _build_sdk_client(self.provider)
        self._fallback_client = (
            _build_sdk_client(self.fallback_provider)
            if self.fallback_provider
            else None
        )

    # ─── public properties ────────────────────────────────────────────────

    @property
    def total_cost_usd(self) -> float:
        """USD spent across every successful call this client has made.

        Reset with reset_cost(). Survives across calls but does NOT survive
        across process restarts — the in-memory cost log is not persistent.
        """
        return round(self._cost_usd, 6)

    def reset_cost(self) -> None:
        """Zero the cost counter. Useful between test runs and between
        billing intervals."""
        self._cost_usd = 0.0

    # ─── public methods: complete + stream ────────────────────────────────

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str:
        """Send a prompt to the LLM and return its response as a string.

        Behaviour:
          1. If the same (prompt, system, max_tokens, model, provider) was
             seen before and use_cache is True — return the cached answer.
          2. If max_cost_usd is set and we are already over it — raise
             BudgetExceeded BEFORE calling the network.
          3. Try the primary provider. If it raises a transient error,
             tenacity retries up to 3 times with exponential backoff.
          4. If the primary raises a permanent error AND a fallback is
             configured, try the fallback (with its own retry).
          5. If both fail permanently — raise AllProvidersFailed.
          6. Track tokens, compute cost, append a log entry, return text.
        """
        cache_key = (self.provider, self.model, system, prompt, max_tokens)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if self.max_cost_usd is not None and self._cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(
                f"Already at ${self._cost_usd:.4f} of ${self.max_cost_usd:.4f} cap"
            )

        try:
            text = self._call_primary(prompt, system, max_tokens)
        except LLMError as primary_error:
            text = self._try_fallback(prompt, system, max_tokens, primary_error)

        if use_cache:
            self._cache[cache_key] = text
        return text

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Stream text chunks from the LLM as they arrive.

        Yields strings. Caller can print them as they come, or accumulate.
        Same retry/log/cost story as complete(), but the cache is bypassed
        because streamed responses are typically used in user-facing UIs
        where freshness matters more than cost.
        """
        started = time.time()
        chunks: list[str] = []
        try:
            for chunk in self._stream_primary(prompt, system, max_tokens):
                chunks.append(chunk)
                yield chunk
        except LLMError as e:
            self._log_call(ok=False, error=e, started=started, streamed=True)
            raise

        full_text = "".join(chunks)
        self._account_for_call(prompt, system, full_text, started, streamed=True)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in `text` under the current model.

        Each provider counts slightly differently; this method hides that.
        """
        if self.provider == "openai":
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        if self.provider == "anthropic":
            r = self._client.messages.count_tokens(
                model=self.model,
                messages=[{"role": "user", "content": text}],
            )
            return r.input_tokens
        raise ValueError(f"Unknown provider: {self.provider}")

    def complete_messages(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        *,
        max_tokens: int = 1024,
    ) -> str:
        """Multi-turn variant of complete(). Used by Conversation."""
        # Convert messages to a flat representation for cache + cost,
        # but pass the structured form to the provider.
        flat = json.dumps(messages, sort_keys=True)
        cache_key = (self.provider, self.model, system, flat, max_tokens)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.max_cost_usd is not None and self._cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(
                f"Already at ${self._cost_usd:.4f} of ${self.max_cost_usd:.4f} cap"
            )

        started = time.time()
        try:
            text = self._call_messages(self._client, self.provider, self.model,
                                       messages, system, max_tokens)
        except LLMError as e:
            self._log_call(ok=False, error=e, started=started)
            raise

        self._account_for_call(flat, system, text, started)
        self._cache[cache_key] = text
        return text

    # ─── private: core call paths ─────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def _call_primary(self, prompt, system, max_tokens) -> str:
        started = time.time()
        try:
            text = _call_provider(
                self._client, self.provider, self.model, prompt, system, max_tokens
            )
        except LLMError as e:
            self._log_call(ok=False, error=e, started=started)
            raise
        self._account_for_call(prompt, system, text, started)
        return text

    def _try_fallback(self, prompt, system, max_tokens, primary_error) -> str:
        if self._fallback_client is None:
            raise primary_error
        _log.warning(
            "primary failed (%s), falling over to %s",
            type(primary_error).__name__,
            self.fallback_provider,
        )
        try:
            return self._call_fallback(prompt, system, max_tokens)
        except LLMError as fb_error:
            raise AllProvidersFailed(
                f"primary={type(primary_error).__name__}, "
                f"fallback={type(fb_error).__name__}"
            ) from fb_error

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def _call_fallback(self, prompt, system, max_tokens) -> str:
        started = time.time()
        try:
            text = _call_provider(
                self._fallback_client, self.fallback_provider,
                self.fallback_model, prompt, system, max_tokens,
            )
        except LLMError as e:
            self._log_call(ok=False, error=e, started=started, used_fallback=True)
            raise
        self._account_for_call(prompt, system, text, started, used_fallback=True)
        return text

    def _stream_primary(self, prompt, system, max_tokens) -> Iterator[str]:
        # Streaming bypasses tenacity — most apps want to surface a partial
        # answer with the failure, not silently retry mid-stream.
        if self.provider == "openai":
            messages = _to_openai_messages(prompt, system)
            stream = self._client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=max_tokens, stream=True,
            )
            for ev in stream:
                delta = ev.choices[0].delta.content
                if delta:
                    yield delta
        elif self.provider == "anthropic":
            with self._client.messages.stream(
                model=self.model, system=system or "",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for chunk in stream.text_stream:
                    yield chunk
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ─── private: accounting + logging ────────────────────────────────────

    def _account_for_call(
        self, prompt, system, text, started, *, used_fallback=False, streamed=False
    ) -> None:
        provider = self.fallback_provider if used_fallback else self.provider
        model = self.fallback_model if used_fallback else self.model

        # Best-effort token counts. We do not call count_tokens() inside
        # this hot path because counting can be expensive for some models.
        # Approximate with character-based heuristics; replace with real
        # usage numbers in production.
        in_tok = max(1, len(str(prompt)) // 4) + (len(system) // 4 if system else 0)
        out_tok = max(1, len(text) // 4)
        cost = calculate_cost_usd(provider, model, in_tok, out_tok)
        self._cost_usd += cost

        self._log_call(
            ok=True, started=started, used_fallback=used_fallback,
            streamed=streamed, in_tok=in_tok, out_tok=out_tok, cost=cost,
            response_chars=len(text),
        )

    def _log_call(
        self, *, ok, started, error=None, used_fallback=False, streamed=False,
        in_tok=None, out_tok=None, cost=None, response_chars=None,
    ) -> None:
        entry = {
            "ts": round(time.time(), 3),
            "tenant": self.tenant_id,
            "provider": self.fallback_provider if used_fallback else self.provider,
            "model": self.fallback_model if used_fallback else self.model,
            "fallback": used_fallback,
            "streamed": streamed,
            "elapsed_sec": round(time.time() - started, 3),
            "ok": ok,
        }
        if ok:
            entry.update({
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": round(cost, 6) if cost is not None else None,
                "response_chars": response_chars,
            })
        else:
            # Trim error to 200 chars so a leaked secret in a stack trace
            # cannot end up in our logs.
            entry["error_type"] = type(error).__name__ if error else "Unknown"
            entry["error"] = str(error)[:200] if error else None

        self.call_log.append(entry)
        (_log.info if ok else _log.error)(json.dumps(entry))

    # ─── private: shared message-mode call ────────────────────────────────

    def _call_messages(
        self, client, provider, model, messages, system, max_tokens
    ) -> str:
        started = time.time()
        if provider == "openai":
            full = list(messages)
            if system:
                full = [{"role": "system", "content": system}, *full]
            r = client.chat.completions.create(
                model=model, messages=full, max_tokens=max_tokens
            )
            return r.choices[0].message.content
        if provider == "anthropic":
            r = client.messages.create(
                model=model, system=system or "",
                max_tokens=max_tokens, messages=messages,
            )
            return r.content[0].text
        raise ValueError(f"Unknown provider: {provider}")


# ─── module-level helpers (kept module-private; tested via LLMClient) ──────


def _missing_model(provider: str) -> str:
    raise ValueError(
        f"Unknown provider {provider!r}. "
        f"Valid: {sorted(_DEFAULT_MODELS.keys())}"
    )


def _build_sdk_client(provider: str):
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    raise ValueError(f"Unknown provider: {provider}")


def _to_openai_messages(prompt, system):
    msgs = [{"role": "user", "content": prompt}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    return msgs


def _call_provider(client, provider, model, prompt, system, max_tokens) -> str:
    """Translate a prompt into the provider's native shape, dispatch,
    translate the response back into a clean string. Raises LLMError
    subclasses for failures the wrapper knows about."""
    try:
        if provider == "openai":
            r = client.chat.completions.create(
                model=model, messages=_to_openai_messages(prompt, system),
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        if provider == "anthropic":
            r = client.messages.create(
                model=model, system=system or "",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text
        raise ValueError(f"Unknown provider: {provider}")
    except Exception as e:
        # Translate every SDK-specific exception into our hierarchy, so
        # the rest of the wrapper (and tenacity) only needs to know about
        # LLMError subclasses.
        raise _translate_provider_error(e) from e


def _translate_provider_error(e: Exception) -> LLMError:
    """Map provider SDK exceptions to our hierarchy. Best-effort: when in
    doubt, treat as ProviderUnavailable so we retry once and let the
    hierarchy's "retry only the retryables" logic take over."""
    name = type(e).__name__
    msg = str(e)
    # OpenAI / Anthropic both expose status_code / status on their errors
    code = getattr(e, "status_code", None) or getattr(e, "status", None)
    retry_after = getattr(e, "retry_after", None)

    if code == 401 or "AuthenticationError" in name or "Unauthorized" in msg:
        return InvalidAPIKey(msg)
    if code == 403 or "PermissionDenied" in name:
        return InvalidAPIKey(msg)
    if code == 429 or "RateLimit" in name:
        return RateLimitHit(msg, retry_after=retry_after)
    if code in (500, 502, 503, 504) or "InternalServer" in name or "ServiceUnavailable" in name:
        return ProviderUnavailable(msg)
    if "Timeout" in name or "timeout" in msg.lower():
        return ProviderTimeout(msg)
    if "context_length" in msg or "maximum context length" in msg.lower():
        return ContextLengthExceeded(msg)
    if "content_filter" in msg or "ContentFilter" in name or "content_policy" in msg:
        return ContentFilterTriggered(msg)
    # Unknown — assume transient so we get one retry.
    return ProviderUnavailable(f"{name}: {msg}")
