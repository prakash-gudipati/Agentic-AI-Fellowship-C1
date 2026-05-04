"""
LLMClient — Session 16 production-grade wrapper (extends S15 v2).

Four new capabilities layered ONTO the S15 wrapper, with the same shell:

  1. PROMPT CACHING            — System prompts are cached at the provider
     (cache_system=True default)  side. Anthropic uses cache_control blocks;
                                  OpenAI uses automatic prefix caching.
                                  Effect: cached input tokens cost ~10% of
                                  uncached input tokens. Big system prompts
                                  on repeat calls drop to roughly zero cost.

  2. NATIVE STRUCTURED OUTPUTS — complete_structured(prompt, schema)
     replaces the S14 ask-for-JSON  uses OpenAI strict mode + Anthropic
     pattern.                       forced tool_use. The model literally
                                  cannot return invalid JSON.

  3. REASONING MODEL ROUTING   — complete_reasoning(prompt, effort=…) routes
                                  to o1 / extended-thinking models. Different
                                  prompt rules (no chain-of-thought; state
                                  goals + constraints), longer timeout,
                                  separate retry policy.

  4. INDIRECT INJECTION DEFENSE — safe_complete(prompt, untrusted_documents=…)
     (the RAG footgun)           wraps every untrusted document in
                                  spotlight tags, sanitises user input,
                                  filters output. PromptInjectionDetected
                                  on either end fails fast.

The S15 methods (complete, stream, complete_messages, count_tokens) all
still work exactly as before — students extend their S15 wrapper, they
do not rewrite it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Iterator, Optional, Type, TypeVar

from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)

from .cost import calculate_cost_usd
from .errors import (
    AllProvidersFailed, BudgetExceeded, ContentFilterTriggered,
    ContextLengthExceeded, InvalidAPIKey, LLMError,
    PromptInjectionDetected, ProviderTimeout, ProviderUnavailable,
    RateLimitHit, ReasoningTimeout, StructuredOutputFailed,
)
from .security import (
    output_filter, safe_combine_documents, sanitize_input,
    spotlight_system_instruction,
)
from .structured import (
    parse_response, parse_tool_input,
    to_anthropic_tool, to_openai_response_format,
)

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # structured methods will raise if pydantic missing

T = TypeVar("T")

_log = logging.getLogger("llm_client")


_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
}

# Routing table for reasoning calls — primary picks the right thinking model
_REASONING_MODELS = {
    "anthropic": "claude-sonnet-4-6",   # used with extended_thinking
    "openai": "o1-mini",                # o1 family
}

_RETRYABLE = (ProviderTimeout, ProviderUnavailable, RateLimitHit)
_REASONING_RETRYABLE = (ReasoningTimeout, ProviderUnavailable, RateLimitHit)


class LLMClient:
    """Reusable, provider-agnostic LLM client with retries, fallback,
    structured logging, cost tracking, streaming, prompt caching,
    structured outputs, reasoning routing, and injection defense."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        *,
        fallback_provider: Optional[str] = None,
        fallback_model: Optional[str] = None,
        tenant_id: Optional[str] = None,
        max_cost_usd: Optional[float] = None,
        # S16 NEW — default ON. Set False to disable provider-side caching.
        cache_system_prompts: bool = True,
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
        self.cache_system_prompts = cache_system_prompts

        self.call_log: list[dict] = []
        self._cache: dict = {}
        self._cost_usd = 0.0

        self._client = _build_sdk_client(self.provider)
        self._fallback_client = (
            _build_sdk_client(self.fallback_provider) if self.fallback_provider else None
        )

    # ─── public properties ────────────────────────────────────────────────

    @property
    def total_cost_usd(self) -> float:
        return round(self._cost_usd, 6)

    def reset_cost(self) -> None:
        self._cost_usd = 0.0

    # ─── S15 method: complete (now with prompt caching) ──────────────────

    def complete(
        self, prompt: str, system: Optional[str] = None, *,
        max_tokens: int = 1024, use_cache: bool = True,
    ) -> str:
        """Send a prompt and return the response as a string.

        S16 change: when cache_system_prompts is True (default) AND a
        system prompt is provided, the wrapper marks the system prompt
        as cacheable on Anthropic (via cache_control) and benefits from
        OpenAI's automatic prefix caching. The cost log distinguishes
        cached vs uncached input tokens.
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

    # ─── S16 NEW: complete_structured ────────────────────────────────────

    def complete_structured(
        self, prompt: str, schema: "Type[BaseModel]", *,
        system: Optional[str] = None, max_tokens: int = 1024,
    ):
        """Force the model to return a Pydantic-validated object.

        Returns an instance of `schema`. Raises StructuredOutputFailed
        if the response cannot be validated. With strict mode this is
        very rare — the API guarantees the JSON shape.

        Cache and budget guard work the same as complete().
        """
        if BaseModel is None:
            raise ImportError("pydantic is required for complete_structured()")

        cache_key = (
            "structured", self.provider, self.model, system, prompt,
            max_tokens, schema.__name__,
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        if self.max_cost_usd is not None and self._cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(
                f"Already at ${self._cost_usd:.4f} of ${self.max_cost_usd:.4f} cap"
            )

        started = time.time()
        try:
            obj = self._call_structured(prompt, system, max_tokens, schema)
        except LLMError as e:
            self._log_call(ok=False, error=e, started=started, kind="structured")
            raise
        # Account for the call. We pass the JSON form for token counting.
        text_estimate = obj.model_dump_json()
        self._account_for_call(prompt, system, text_estimate, started, kind="structured")
        self._cache[cache_key] = obj
        return obj

    # ─── S16 NEW: complete_reasoning ─────────────────────────────────────

    def complete_reasoning(
        self, prompt: str, *,
        system: Optional[str] = None, effort: str = "medium",
        max_tokens: int = 4096,
    ) -> str:
        """Route to a reasoning model (o1, extended-thinking) for hard
        multi-step problems.

        Prompting rules differ from regular models:
          - DO state the goal and constraints clearly.
          - DO NOT add 'think step by step' — reasoning models already do.
          - DO NOT add few-shot examples — they hurt more than they help
            on reasoning models per provider guidance.
          - DO allow more max_tokens — reasoning tokens count.

        effort: "low" | "medium" | "high" — passed to OpenAI o-series as
        reasoning_effort; mapped to extended_thinking budget on Anthropic.
        """
        if self.max_cost_usd is not None and self._cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(
                f"Already at ${self._cost_usd:.4f} of ${self.max_cost_usd:.4f} cap"
            )

        started = time.time()
        try:
            text = self._call_reasoning(prompt, system, max_tokens, effort)
        except LLMError as e:
            self._log_call(ok=False, error=e, started=started, kind="reasoning")
            raise
        self._account_for_call(prompt, system, text, started, kind="reasoning",
                               model_override=_REASONING_MODELS.get(self.provider))
        return text

    # ─── S16 NEW: safe_complete (RAG-safe) ───────────────────────────────

    def safe_complete(
        self, prompt: str,
        untrusted_documents: list[str], *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        document_tag: str = "document",
    ) -> str:
        """Run a normal complete() but with full prompt-injection defense.

        Three defense layers, in order:
          1. Sanitize the user's own prompt (raises PromptInjectionDetected
             on a known pattern).
          2. Spotlight every untrusted document by wrapping it in
             <document>…</document> tags AND prepending a system-prompt
             instruction telling the model to treat tagged content as
             data, not instructions.
          3. After the model responds, scan the output for evidence of
             injection success (system-prompt leak, jailbreak claim).

        Use this method whenever the prompt context contains text that
        did not come from your own developers — RAG retrievals, user
        uploads, web scrapes, support tickets. From Phase 4 onwards in
        this fellowship, all RAG generation steps go through this path.
        """
        # Layer 1
        sanitize_input(prompt)

        # Layer 2
        spotlit = safe_combine_documents(untrusted_documents, tag=document_tag)
        defense = spotlight_system_instruction(tag=document_tag)
        full_system = (system + "\n\n" + defense) if system else defense
        full_prompt = f"{prompt}\n\nReference material:\n{spotlit}"

        # Run the normal cached/retry/fallback path
        response = self.complete(
            full_prompt, system=full_system, max_tokens=max_tokens, use_cache=False
        )

        # Layer 3
        output_filter(response)  # raises PromptInjectionDetected on match

        return response

    # ─── S15 methods (unchanged): stream, count_tokens, complete_messages ─

    def stream(
        self, prompt: str, system: Optional[str] = None, *,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        started = time.time()
        chunks: list[str] = []
        try:
            for chunk in self._stream_primary(prompt, system, max_tokens):
                chunks.append(chunk)
                yield chunk
        except LLMError as e:
            self._log_call(ok=False, error=e, started=started, streamed=True)
            raise
        full = "".join(chunks)
        self._account_for_call(prompt, system, full, started, streamed=True)

    def count_tokens(self, text: str) -> int:
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

    # ─── private: structured-output call paths ───────────────────────────

    def _call_structured(self, prompt, system, max_tokens, schema):
        if self.provider == "openai":
            messages = [{"role": "user", "content": prompt}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            try:
                r = self._client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=max_tokens,
                    response_format=to_openai_response_format(schema),
                )
                return parse_response(r.choices[0].message.content, schema)
            except Exception as e:
                raise _translate_provider_error(e) from e
        if self.provider == "anthropic":
            tool = to_anthropic_tool(schema)
            try:
                r = self._client.messages.create(
                    model=self.model, system=_apply_anthropic_cache(system, self.cache_system_prompts),
                    max_tokens=max_tokens,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool["name"]},
                    messages=[{"role": "user", "content": prompt}],
                )
                # Anthropic returns a list of content blocks; find the tool_use
                for block in r.content:
                    if getattr(block, "type", None) == "tool_use":
                        return parse_tool_input(block.input, schema)
                raise StructuredOutputFailed("Anthropic returned no tool_use block")
            except Exception as e:
                if isinstance(e, LLMError):
                    raise
                raise _translate_provider_error(e) from e
        raise ValueError(f"Unknown provider: {self.provider}")

    # ─── private: reasoning call paths ───────────────────────────────────

    @retry(
        stop=stop_after_attempt(2),  # reasoning calls are expensive — fewer retries
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type(_REASONING_RETRYABLE),
        reraise=True,
    )
    def _call_reasoning(self, prompt, system, max_tokens, effort):
        model = _REASONING_MODELS.get(self.provider)
        if not model:
            raise ValueError(f"No reasoning model registered for {self.provider}")
        try:
            if self.provider == "openai":
                messages = [{"role": "user", "content": prompt}]
                # Note: o1 series does not accept system messages via the
                # role; concatenate into the user message if needed.
                if system:
                    messages[0]["content"] = system + "\n\n" + prompt
                r = self._client.chat.completions.create(
                    model=model, messages=messages,
                    max_completion_tokens=max_tokens,
                    reasoning_effort=effort,
                )
                return r.choices[0].message.content
            if self.provider == "anthropic":
                # Extended thinking — set thinking budget by effort
                budget = {"low": 1024, "medium": 4096, "high": 16000}.get(effort, 4096)
                r = self._client.messages.create(
                    model=model,
                    system=_apply_anthropic_cache(system or "", self.cache_system_prompts),
                    max_tokens=max(max_tokens, budget + 1024),
                    thinking={"type": "enabled", "budget_tokens": budget},
                    messages=[{"role": "user", "content": prompt}],
                )
                # Skip thinking block, return the actual text block
                for block in r.content:
                    if getattr(block, "type", None) == "text":
                        return block.text
                raise StructuredOutputFailed("Anthropic reasoning returned no text block")
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            translated = _translate_provider_error(e)
            # Reasoning calls map plain timeouts onto ReasoningTimeout
            if isinstance(translated, ProviderTimeout):
                raise ReasoningTimeout(str(e)) from e
            raise translated from e
        raise ValueError(f"Unknown provider: {self.provider}")

    # ─── private: regular complete call paths (S15 plus cache_control) ───

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
                self._client, self.provider, self.model, prompt,
                system, max_tokens, cache_system=self.cache_system_prompts,
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
            type(primary_error).__name__, self.fallback_provider,
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
                cache_system=self.cache_system_prompts,
            )
        except LLMError as e:
            self._log_call(ok=False, error=e, started=started, used_fallback=True)
            raise
        self._account_for_call(prompt, system, text, started, used_fallback=True)
        return text

    def _stream_primary(self, prompt, system, max_tokens) -> Iterator[str]:
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
                model=self.model,
                system=_apply_anthropic_cache(system or "", self.cache_system_prompts),
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for chunk in stream.text_stream:
                    yield chunk
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ─── private: accounting + logging (S15 plus cached_input_tokens) ────

    def _account_for_call(
        self, prompt, system, text, started, *,
        used_fallback=False, streamed=False, kind="complete",
        model_override=None,
    ) -> None:
        provider = self.fallback_provider if used_fallback else self.provider
        model = model_override or (
            self.fallback_model if used_fallback else self.model
        )

        in_tok = max(1, len(str(prompt)) // 4) + (len(system) // 4 if system else 0)
        out_tok = max(1, len(text) // 4)
        # On a cached call (cache_system_prompts True AND a system prompt
        # was supplied), assume the system portion is a cache hit on the
        # second-and-subsequent calls. Real production reads the actual
        # cached_tokens from the API response — we estimate.
        cached = (len(system) // 4) if (self.cache_system_prompts and system) else 0
        cost = calculate_cost_usd(provider, model, in_tok, out_tok, cached)
        self._cost_usd += cost

        self._log_call(
            ok=True, started=started, used_fallback=used_fallback,
            streamed=streamed, kind=kind,
            in_tok=in_tok, cached_in_tok=cached, out_tok=out_tok,
            cost=cost, response_chars=len(text), model_override=model_override,
        )

    def _log_call(
        self, *, ok, started, error=None, used_fallback=False, streamed=False,
        kind="complete", in_tok=None, cached_in_tok=None, out_tok=None,
        cost=None, response_chars=None, model_override=None,
    ) -> None:
        entry = {
            "ts": round(time.time(), 3),
            "tenant": self.tenant_id,
            "provider": self.fallback_provider if used_fallback else self.provider,
            "model": model_override or (
                self.fallback_model if used_fallback else self.model
            ),
            "kind": kind,
            "fallback": used_fallback,
            "streamed": streamed,
            "elapsed_sec": round(time.time() - started, 3),
            "ok": ok,
        }
        if ok:
            entry.update({
                "input_tokens": in_tok,
                "cached_input_tokens": cached_in_tok,
                "output_tokens": out_tok,
                "cost_usd": round(cost, 6) if cost is not None else None,
                "response_chars": response_chars,
            })
        else:
            entry["error_type"] = type(error).__name__ if error else "Unknown"
            entry["error"] = str(error)[:200] if error else None
        self.call_log.append(entry)
        (_log.info if ok else _log.error)(json.dumps(entry))


# ─── module-level helpers ─────────────────────────────────────────────────


def _missing_model(provider: str) -> str:
    raise ValueError(
        f"Unknown provider {provider!r}. Valid: {sorted(_DEFAULT_MODELS.keys())}"
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


def _apply_anthropic_cache(system: str, enabled: bool):
    """Return the Anthropic 'system' parameter, optionally with cache_control.

    When enabled and the system prompt is non-trivial (≥1024 chars),
    return the structured form: a list with one dict carrying
    cache_control: {"type": "ephemeral"}. Anthropic caches that block
    server-side. Subsequent calls with the same system text get the
    90% input-cost discount on cached tokens.

    For shorter system prompts the discount is not worth the extra
    request shape — pass plain string.
    """
    if not system:
        return ""
    if not enabled or len(system) < 1024:
        return system
    return [{
        "type": "text",
        "text": system,
        "cache_control": {"type": "ephemeral"},
    }]


def _call_provider(client, provider, model, prompt, system, max_tokens,
                   *, cache_system: bool = False) -> str:
    """Provider dispatch — same as S15, with cache_control plumbed in for
    Anthropic and a no-op for OpenAI (it caches automatically based on
    prompt prefix length, no special API call needed)."""
    try:
        if provider == "openai":
            r = client.chat.completions.create(
                model=model,
                messages=_to_openai_messages(prompt, system),
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        if provider == "anthropic":
            r = client.messages.create(
                model=model,
                system=_apply_anthropic_cache(system or "", cache_system),
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text
        raise ValueError(f"Unknown provider: {provider}")
    except Exception as e:
        raise _translate_provider_error(e) from e


def _translate_provider_error(e: Exception) -> LLMError:
    """Map provider SDK exceptions to our hierarchy."""
    name = type(e).__name__
    msg = str(e)
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
    return ProviderUnavailable(f"{name}: {msg}")
