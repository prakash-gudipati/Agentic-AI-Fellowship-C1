"""
Demo 1 — Swap providers with one line.

Runs the same prompt against Anthropic and then OpenAI, prints both
responses, prints the per-call log entries (provider, model, tokens,
cost, latency, ok), and prints the running cost.

Run:
    $ python -m examples.swap_providers_demo
"""

import json
from dotenv import load_dotenv
load_dotenv()

from llm_client import LLMClient


PROMPT = "Explain what a vending machine does in one sentence."

for provider in ("anthropic", "openai"):
    print(f"\n=== {provider.upper()} ===")
    client = LLMClient(provider=provider)

    answer = client.complete(PROMPT)
    print(answer)
    print(f"\nTokens in prompt: {client.count_tokens(PROMPT)}")
    print(f"Total cost so far: ${client.total_cost_usd:.6f}")

    print("\nLog entries:")
    for entry in client.call_log:
        print(json.dumps(entry, indent=2))
