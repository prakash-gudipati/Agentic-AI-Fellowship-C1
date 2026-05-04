"""
Tests for LLMClient.

Two kinds of tests live here:
  1. INTEGRATION — real API calls. Marked @pytest.mark.live. Cost real
     money (cents). Run sparingly: `pytest -m live`.
  2. UNIT — every other test. Mock the SDK; never touch the network.
     Fast, deterministic, run on every CI build.

Why test a wrapper at all? Because the wrapper IS the abstraction the
rest of the application depends on. If the wrapper has a bug — every
caller has the bug. Tests pin down the contract.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from llm_client import LLMClient, Conversation
import llm_client.client as client_module
from llm_client.errors import (
    AllProvidersFailed,
    BudgetExceeded,
    InvalidAPIKey,
    LLMError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimitHit,
)


# ─── exception translation ────────────────────────────────────────────────


def test_translate_401_becomes_invalid_api_key():
    e = Exception("AuthenticationError: bad key")
    e.status_code = 401
    out = client_module._translate_provider_error(e)
    assert isinstance(out, InvalidAPIKey)


def test_translate_429_becomes_rate_limit_hit_with_retry_after():
    e = Exception("RateLimitError")
    e.status_code = 429
    e.retry_after = 12.5
    out = client_module._translate_provider_error(e)
    assert isinstance(out, RateLimitHit)
    assert out.retry_after == 12.5


def test_translate_503_becomes_provider_unavailable():
    e = Exception("ServiceUnavailable")
    e.status_code = 503
    out = client_module._translate_provider_error(e)
    assert isinstance(out, ProviderUnavailable)


def test_unknown_error_becomes_provider_unavailable():
    e = Exception("Some weird new error")
    out = client_module._translate_provider_error(e)
    # Conservative: assume transient so we get one retry, then propagate.
    assert isinstance(out, ProviderUnavailable)


# ─── retry behaviour ──────────────────────────────────────────────────────


def test_does_not_retry_on_invalid_api_key(monkeypatch):
    """A bad key is permanent. Retrying is pointless and looks like an
    attack. We must NOT retry."""
    call_count = {"n": 0}

    def fail_fast(*args, **kwargs):
        call_count["n"] += 1
        raise InvalidAPIKey("bad key")

    monkeypatch.setattr(client_module, "_call_provider", fail_fast)
    os.environ["ANTHROPIC_API_KEY"] = "test-fake-key"
    client = LLMClient(provider="anthropic")

    with pytest.raises(InvalidAPIKey):
        client.complete("hi")

    assert call_count["n"] == 1, "InvalidAPIKey must NOT trigger retries"


def test_retries_on_provider_unavailable(monkeypatch):
    """503-class errors are transient — wrapper should retry up to 3 times."""
    call_count = {"n": 0}

    def transient(*args, **kwargs):
        call_count["n"] += 1
        raise ProviderUnavailable("simulated 503")

    monkeypatch.setattr(client_module, "_call_provider", transient)
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_: None)  # skip waits in tests
    os.environ["ANTHROPIC_API_KEY"] = "test-fake-key"
    client = LLMClient(provider="anthropic")

    with pytest.raises(ProviderUnavailable):
        client.complete("hi")

    assert call_count["n"] == 3, f"expected 3 retry attempts, got {call_count['n']}"


# ─── fallback chain ───────────────────────────────────────────────────────


def test_fallback_kicks_in_when_primary_permanent_failure(monkeypatch):
    """Primary fails 3 times → wrapper falls over to secondary, which
    succeeds. End-user gets the secondary's answer."""
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_: None)

    def router(sdk, provider, model, prompt, system, max_tokens):
        if provider == "anthropic":
            raise ProviderUnavailable("anthropic is down")
        if provider == "openai":
            return f"Hello from openai/{model}"
        raise AssertionError("unexpected provider")

    monkeypatch.setattr(client_module, "_call_provider", router)
    os.environ["ANTHROPIC_API_KEY"] = "test-fake"
    os.environ["OPENAI_API_KEY"] = "test-fake"

    client = LLMClient(provider="anthropic", fallback_provider="openai")
    out = client.complete("hi")

    assert "openai" in out, "fallback path did not return"
    assert any(e.get("fallback") for e in client.call_log), "no fallback log entry"


def test_all_providers_failed_raises_when_both_die(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda *_: None)

    def both_die(sdk, provider, model, *args, **kwargs):
        raise ProviderUnavailable(f"{provider} is down")

    monkeypatch.setattr(client_module, "_call_provider", both_die)
    os.environ["ANTHROPIC_API_KEY"] = "test-fake"
    os.environ["OPENAI_API_KEY"] = "test-fake"

    client = LLMClient(provider="anthropic", fallback_provider="openai")
    with pytest.raises(AllProvidersFailed):
        client.complete("hi")


# ─── budget guard ─────────────────────────────────────────────────────────


def test_budget_exceeded_raises_before_calling_network(monkeypatch):
    """If we are already at the cost cap, the wrapper must NOT make the
    network call. The whole point of a budget guard is preventing the
    spend, not reporting it after."""
    network_called = {"yes": False}

    def fake_call(*args, **kwargs):
        network_called["yes"] = True
        return "should not happen"

    monkeypatch.setattr(client_module, "_call_provider", fake_call)
    os.environ["ANTHROPIC_API_KEY"] = "test-fake"

    client = LLMClient(provider="anthropic", max_cost_usd=0.001)
    client._cost_usd = 0.005   # already over the cap

    with pytest.raises(BudgetExceeded):
        client.complete("hi")

    assert not network_called["yes"], "wrapper called the network despite cost cap"


# ─── caching ──────────────────────────────────────────────────────────────


def test_cache_returns_same_result_without_calling_again(monkeypatch):
    call_count = {"n": 0}

    def stub(*args, **kwargs):
        call_count["n"] += 1
        return "cached answer"

    monkeypatch.setattr(client_module, "_call_provider", stub)
    os.environ["ANTHROPIC_API_KEY"] = "test-fake"

    client = LLMClient(provider="anthropic")
    a = client.complete("hi")
    b = client.complete("hi")

    assert a == b == "cached answer"
    assert call_count["n"] == 1, "cache did not prevent the second network call"


# ─── conversation helper ──────────────────────────────────────────────────


def test_conversation_appends_user_and_assistant_each_turn(monkeypatch):
    monkeypatch.setattr(client_module, "_call_provider", lambda *a, **k: "ok")
    # Conversation uses _call_messages, not _call_provider, so we stub the
    # SDK call inside the LLMClient instance directly.
    os.environ["ANTHROPIC_API_KEY"] = "test-fake"
    client = LLMClient(provider="anthropic")

    def fake_messages(self, c, p, m, msgs, system, mt):
        return f"reply-{len(msgs)}"

    monkeypatch.setattr(LLMClient, "_call_messages", fake_messages)

    chat = Conversation(client, system="be brief")
    chat.say("hi")
    chat.say("again")

    assert chat.turn_count() == 2
    assert len(chat.messages) == 4   # user, assistant, user, assistant
    assert chat.messages[0]["role"] == "user"
    assert chat.messages[1]["role"] == "assistant"


# ─── live tests (real API; marked) ────────────────────────────────────────


@pytest.mark.live
def test_live_anthropic_returns_string():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("no anthropic key")
    client = LLMClient(provider="anthropic")
    out = client.complete("Say the single word: ready.")
    assert isinstance(out, str) and len(out) > 0
    assert client.total_cost_usd > 0


@pytest.mark.live
def test_live_openai_returns_string():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("no openai key")
    client = LLMClient(provider="openai")
    out = client.complete("Say the single word: ready.")
    assert isinstance(out, str) and len(out) > 0
