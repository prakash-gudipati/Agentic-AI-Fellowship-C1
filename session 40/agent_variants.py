"""Session 41 (merged into S40) — named AGENT VARIANTS for eval-driven development.

EDD needs two agents that score DIFFERENTLY on the SAME dataset, deterministically
offline, so we can show red->green and A/B comparison. A variant models a "prompt
change" or "code change" you are evaluating BEFORE you ship it.

The mechanism (WHY it works, deterministically, with NO API key):
  The verified FAKE_LLM grounding gate in llm_client.py BLOCKs any answer whose
  number is absent from the KB context, and any answer carrying a [HALLUCINATION]
  marker. The V1_BUGGY variant models a regression — a prompt tweak that made the
  agent hallucinate a settlement figure on ONE question. We simulate that bug WITHOUT
  touching the verified llm_client: the wrapper injects a [HALLUCINATION] marker into
  exactly ONE targeted question before delegating to the real SafeAgent. The grounding
  gate then fires, the answer is BLOCKed, and because that case is must_block=False the
  eval coverage assertion FAILS for V1. V2_FIXED injects nothing, so the same case is
  answered cleanly and PASSES. Same dataset, same harness, two scores — that's the A/B.
"""
from __future__ import annotations

from typing import Dict, Optional

from guardrail_chain import GuardrailChain
from input_guardrails import (InjectionGuardrail, LengthGuardrail,
                              PIIGuardrail, TopicGuardrail)
from output_guardrails import (GroundingGuardrail, PIIRedactionGuardrail,
                               SafetyGuardrail, SchemaGuardrail)
from safe_agent import SafeAgent, DEFAULT_KB


def _input_chain() -> GuardrailChain:
    """Standard input chain — same as demo.py's production default."""
    return GuardrailChain(
        [PIIGuardrail(), InjectionGuardrail(), TopicGuardrail(), LengthGuardrail(2000)],
        stage="input",
    )


def _output_chain() -> GuardrailChain:
    """Standard output chain — redact, shape, safety judge, grounding judge."""
    return GuardrailChain(
        [PIIRedactionGuardrail(), SchemaGuardrail(), SafetyGuardrail(), GroundingGuardrail()],
        stage="output",
    )


# The ONE question the v1 bug hallucinates on. Picked from the GOLDEN set (g3).
BUGGY_QUESTION = "How long do transfers take to settle?"


class VariantAgent:
    """A thin wrapper over SafeAgent that models one named agent CONFIGURATION.

    Same guardrails, same model seam — the ONLY difference is an optional, targeted
    bug injection. This is how we get two deterministic scores from one harness.
    """

    def __init__(self, name: str, buggy: bool = False, kb=None):
        self.name = name
        self.buggy = buggy
        self._agent = SafeAgent(_input_chain(), _output_chain(),
                                kb=kb if kb is not None else DEFAULT_KB)

    def ask(self, user_input: str) -> Dict:
        """Delegate to the real SafeAgent; if buggy, hallucinate on the one case."""
        if self.buggy and user_input.strip() == BUGGY_QUESTION:
            # Models a prompt regression: the agent invents a settlement figure.
            # The verified grounding gate will catch it -> the case fails the suite.
            user_input = "[HALLUCINATION] " + user_input
        return self._agent.ask(user_input)


def make_variants(kb=None) -> Dict[str, VariantAgent]:
    """The two named variants every EDD demo compares: v1 (buggy) vs v2 (fixed)."""
    return {
        "V1_BUGGY": VariantAgent("V1_BUGGY", buggy=True, kb=kb),
        "V2_FIXED": VariantAgent("V2_FIXED", buggy=False, kb=kb),
    }
