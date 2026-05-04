"""
Cost tracking for LLMClient.

Pricing data is hard-coded here for simplicity. In production this lives
in a config file or a small table the wrapper loads at start time. The
shape is the same: per (provider, model), an input rate and an output
rate, both in USD per million tokens.

Update this file when providers change their pricing — the rest of the
wrapper does not change.
"""

from __future__ import annotations


# USD per 1,000,000 tokens. Public list prices as of the v2.1 build.
USD_PER_MTOK: dict[tuple[str, str], dict[str, float]] = {
    # Anthropic
    ("anthropic", "claude-haiku-4-5"):   {"input": 1.00,  "output": 5.00},
    ("anthropic", "claude-sonnet-4-6"):  {"input": 3.00,  "output": 15.00},
    ("anthropic", "claude-opus-4-6"):    {"input": 15.00, "output": 75.00},
    # OpenAI
    ("openai", "gpt-4o-mini"):           {"input": 0.15,  "output": 0.60},
    ("openai", "gpt-4o"):                {"input": 2.50,  "output": 10.00},
}


def calculate_cost_usd(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> float:
    """Return the USD cost of one call. Returns 0.0 if the model is unknown.

    A 0.0 cost is a deliberate choice: never raise on a missing entry. We
    log it (the LLMClient does that) so the operator can update the table,
    but a missing entry never breaks a production call.
    """
    rates = USD_PER_MTOK.get((provider, model))
    if rates is None:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input"] + (
        output_tokens / 1_000_000
    ) * rates["output"]
