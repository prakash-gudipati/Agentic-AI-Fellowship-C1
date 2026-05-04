"""
Demo 3 — Reasoning model routing.

Same wrapper, but routes to a thinking model (o1, extended thinking)
for a hard multi-step problem. Compares against the regular model.

Note the prompt — no "think step by step", no few-shot examples. Just
the goal and constraints. Reasoning models do their own thinking.

Run:
    $ python -m examples.reasoning_demo
"""

import time
from dotenv import load_dotenv
load_dotenv()

from llm_client import LLMClient


PROMPT = (
    "A train leaves Mumbai at 14:00 traveling at 80 km/h. Another train "
    "leaves Pune at 14:30 traveling at 60 km/h on the same track in the "
    "opposite direction. Mumbai and Pune are 150 km apart on this track. "
    "When and where do they meet? Show the equations you used."
)

client = LLMClient(provider="anthropic")

print("=== REGULAR MODEL ===")
t0 = time.time()
print(client.complete(PROMPT))
print(f"\n[{time.time() - t0:.2f} sec, cost ${client.total_cost_usd:.6f}]")

client.reset_cost()

print("\n\n=== REASONING MODEL ===")
t0 = time.time()
print(client.complete_reasoning(PROMPT, effort="medium"))
print(f"\n[{time.time() - t0:.2f} sec, cost ${client.total_cost_usd:.6f}]")

print("\n\nSee how the reasoning model takes longer but produces more "
      "rigorous step-by-step working without you asking for it.")
