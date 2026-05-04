"""
Demo 3 — Fallback chain (the wow moment).

Simulates the Anthropic provider being down by injecting a deliberate
ProviderUnavailable exception, then watches the wrapper transparently
fail over to the OpenAI fallback.

Run:
    $ python -m examples.fallback_demo
"""

import json
from unittest.mock import patch
from dotenv import load_dotenv
load_dotenv()

from llm_client import LLMClient
from llm_client.errors import ProviderUnavailable


client = LLMClient(
    provider="anthropic",            # primary
    fallback_provider="openai",      # if primary fails permanently
)

PROMPT = "What is one good use of a vending machine?"


# Intercept the underlying _call_provider to simulate the primary failing.
# In real life, the provider being down does the same thing — the wrapper
# does not care HOW the failure happens, only that it raises one of our
# retryable error types.
import llm_client.client as client_module
real_call = client_module._call_provider


def flaky_primary(sdk_client, provider, model, *args, **kwargs):
    # Anthropic always fails (3 retries, then permanent for the wrapper).
    if provider == "anthropic":
        raise ProviderUnavailable("simulated 503 from anthropic")
    return real_call(sdk_client, provider, model, *args, **kwargs)


with patch.object(client_module, "_call_provider", flaky_primary):
    answer = client.complete(PROMPT)

print("Final answer (from fallback):")
print(answer)

print("\nCall log — read it from top to bottom:")
for entry in client.call_log:
    print(json.dumps({k: entry.get(k) for k in (
        "provider", "model", "fallback", "ok", "error_type", "elapsed_sec"
    )}, indent=2))

print(f"\nTotal cost: ${client.total_cost_usd:.6f} "
      f"(only the fallback call was billed; the failed primary attempts cost $0)")
