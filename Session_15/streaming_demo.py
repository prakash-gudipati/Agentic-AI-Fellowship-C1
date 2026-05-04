"""
Demo 2 — Streaming.

Streams a longer response one chunk at a time and prints each chunk as
it arrives. The end-of-stream summary shows total cost and latency.

Run:
    $ python -m examples.streaming_demo
"""

import sys
import time
from dotenv import load_dotenv
load_dotenv()

from llm_client import LLMClient


PROMPT = (
    "List three things a vending machine and a chatbot have in common. "
    "Use one short paragraph per thing. Reply in 5 paragaraphs."
)

client = LLMClient(provider="anthropic")

print("Streaming response:")
print("-" * 60)
started = time.time()
for chunk in client.stream(PROMPT, max_tokens=400):
    sys.stdout.write(chunk)
    sys.stdout.flush()
elapsed = time.time() - started
print(f"\n{'-' * 60}")
print(f"Streamed in {elapsed:.2f} sec.")
print(f"Total cost: ${client.total_cost_usd:.6f}")
print(f"Calls logged: {len(client.call_log)}")
