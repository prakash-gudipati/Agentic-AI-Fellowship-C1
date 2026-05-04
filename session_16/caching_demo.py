"""
Demo 1 — Prompt caching dropping cost on the second call.

Calls the same long system prompt twice. First call: full input cost.
Second call: cached input portion costs ~10% of normal.

Run:
    $ python -m examples.caching_demo
"""

import json
from dotenv import load_dotenv
load_dotenv()

from llm_client import LLMClient


# A long-ish system prompt — the kind a real product would reuse across
# every call. Caching only kicks in when the system prompt is at least
# ~1024 chars (Anthropic's threshold).
LONG_SYSTEM = (
    "You are a senior support engineer at a SaaS company. " * 80
    + "Be concise. Quote ticket numbers exactly. Never speculate about pricing."
)
PROMPT = "What are the three most common billing questions?"


client = LLMClient(provider="anthropic", cache_system_prompts=True)

print("Call 1 — system prompt is fresh, gets cached server-side:")
client.complete(PROMPT, system=LONG_SYSTEM)
entry1 = client.call_log[-1]
print(json.dumps(entry1, indent=2))

print("\nCall 2 — same system prompt, cache hit expected:")
client.complete(PROMPT + " (different question this time)", system=LONG_SYSTEM)
entry2 = client.call_log[-1]
print(json.dumps(entry2, indent=2))

print(f"\nCall 1 input cost (uncached): ${entry1['cost_usd']:.6f}")
print(f"Call 2 input cost (cached):   ${entry2['cost_usd']:.6f}")
ratio = entry2['cost_usd'] / entry1['cost_usd'] if entry1['cost_usd'] else 1
print(f"Cost ratio: {ratio:.1%}  (target: ~10–25%)")
