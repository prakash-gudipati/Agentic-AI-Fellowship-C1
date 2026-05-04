"""
Cost tracking — Session 16 extension of S15.

S16 adds:
  - CACHED_INPUT_RATE_MULTIPLIER — cached tokens cost ~10% of uncached.
  - calculate_cost_usd() now accepts cached_input_tokens, applies the
    discount, and returns the total cost.
"""

from __future__ import annotations


# Cached prompt tokens (provider prompt-cache hits) cost roughly 10% of
# normal input tokens on both Anthropic and OpenAI. Discount is approx;
# tune per provider if your real bills disagree.
CACHED_INPUT_RATE_MULTIPLIER = 0.10


# USD per 1,000,000 tokens. Update when providers change pricing.
USD_PER_MTOK: dict[tuple[str, str], dict[str, float]] = {
    # Anthropic
    ("anthropic", "claude-haiku-4-5"):    {"input": 1.00,  "output": 5.00},
    ("anthropic", "claude-sonnet-4-6"):   {"input": 3.00,  "output": 15.00},
    ("anthropic", "claude-opus-4-6"):     {"input": 15.00, "output": 75.00},
    # OpenAI standard
    ("openai", "gpt-4o-mini"):            {"input": 0.15,  "output": 0.60},
    ("openai", "gpt-4o"):                 {"input": 2.50,  "output": 10.00},
    # OpenAI reasoning models — output much more expensive (reasoning tokens)
    ("openai", "o1-mini"):                {"input": 1.10,  "output": 4.40},
    ("openai", "o1"):                     {"input": 15.00, "output": 60.00},
}


def calculate_cost_usd(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Return USD cost of one call.

    cached_input_tokens — the SUBSET of input_tokens that were a prompt-
    cache hit. Charged at 10% of the normal input rate. Example: a 5,000-
    token system prompt that is fully cached on the second call costs
    5,000 × $1/Mtok × 0.10 = $0.0005, not $0.005.

    Returns 0.0 for unknown (provider, model). Never raises.
    """
    rates = USD_PER_MTOK.get((provider, model))
    if rates is None:
        return 0.0

    uncached = max(0, input_tokens - cached_input_tokens)
    input_cost = (
        (uncached / 1_000_000) * rates["input"]
        + (cached_input_tokens / 1_000_000) * rates["input"] * CACHED_INPUT_RATE_MULTIPLIER
    )
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    return input_cost + output_cost
