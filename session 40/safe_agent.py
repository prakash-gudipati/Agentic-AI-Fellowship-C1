"""Session 40 — SafeAgent: the support agent wrapped in the safety layer.

The agent itself is trivial (one model call). The VALUE is the sandwich:
    input chain  ->  model  ->  output chain
If the input chain blocks, the model is NEVER called (cheaper + safer).
If the output chain blocks, the user gets a safe fallback, not the raw answer.
This is the whole "production safety layer" the session is about.
"""
from __future__ import annotations

from typing import Dict

from guardrail_chain import GuardrailChain
from prompts import AGENT_SYSTEM
import llm_client

# Tiny hard-coded knowledge base — context strings are drawn from here so the
# grounding gate can distinguish grounded answers from hallucinated ones.
DEFAULT_KB = [
    "The refund window is 14 days from the purchase date.",
    "PayMint support hours are 9am to 9pm IST.",
    "Transfers settle in 2 business days.",
    "PayMint never asks for your full card number.",
]

_SAFE_REFUSAL = "I can't help with that request."
_SAFE_FALLBACK = "I'm not able to share a verified answer to that right now."


class SafeAgent:
    """A PayMint support agent guarded on both the input and output sides."""

    def __init__(self, input_chain: GuardrailChain, output_chain: GuardrailChain, kb=None):
        self.input_chain = input_chain
        self.output_chain = output_chain
        self.kb = kb if kb is not None else DEFAULT_KB

    def ask(self, user_input: str) -> Dict:
        """Run one guarded turn. Returns answer + both reports + model_called."""
        context = "\n".join(self.kb)

        # 1) INPUT gate — block-before-spend.
        input_report = self.input_chain.run(user_input)
        if input_report.blocked:
            return {
                "answer": _SAFE_REFUSAL,
                "input_report": input_report,
                "output_report": None,
                "model_called": False,
            }

        # The input chain may have REDACTed the text — use the cleaned version.
        clean_input = input_report.final_text
        prompt = f"CONTEXT:\n{context}\n\nQUESTION: {clean_input}"

        # 2) MODEL call (only reached if input passed).
        raw_answer = llm_client.complete(AGENT_SYSTEM, prompt)

        # 3) OUTPUT gate — validate before the user sees it.
        output_report = self.output_chain.run(raw_answer, context=context)
        if output_report.blocked:
            return {
                "answer": _SAFE_FALLBACK,
                "input_report": input_report,
                "output_report": output_report,
                "model_called": True,
            }

        return {
            "answer": output_report.final_text,  # may be redacted
            "input_report": input_report,
            "output_report": output_report,
            "model_called": True,
        }
