"""
Session 34 — single_agent_baseline.py

The single-agent baseline. One LLM call. One prompt that asks for
research + writing + verification in one go. Intentionally mediocre —
it's the baseline we beat with the multi-agent crew in Demo 1.

This file is what we are LEAVING. Naive multi-agent is an upgrade.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from llm_client import LLMClient
from prompts import SINGLE_AGENT_BASELINE_SYSTEM


@dataclass
class SingleAgentResult:
    user_request: str
    final_text: str
    elapsed_seconds: float
    llm_calls: int


def answer(
    user_request: str, llm: Optional[LLMClient] = None
) -> SingleAgentResult:
    llm = llm or LLMClient()
    started = time.perf_counter()
    user_prompt = (
        f"TOPIC: {user_request}\n\n"
        "Produce a 3-paragraph brief that does research, writing, and "
        "verification in one response. Cite sources inline."
    )
    text = llm.complete(SINGLE_AGENT_BASELINE_SYSTEM, user_prompt, max_tokens=700)
    elapsed = time.perf_counter() - started
    return SingleAgentResult(
        user_request=user_request,
        final_text=text,
        elapsed_seconds=elapsed,
        llm_calls=llm.call_count,
    )
