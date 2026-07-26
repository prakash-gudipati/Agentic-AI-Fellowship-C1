"""Session 40 — the one LLM seam, with a deterministic OFFLINE fake.

complete(system, user) is the ONLY way this codebase talks to a model.
Real path: anthropic SDK. Fake path (FAKE_LLM=1 or no key): a canned router
that dispatches on the system prompt's OPENER PHRASE ("you are a ...") only.
WHY one seam: the agent and both LLM-as-judge guardrails go through here, so
the whole safety layer runs offline + deterministically for the walkthrough.
"""
from __future__ import annotations

import json
import os
import re


def _use_fake() -> bool:
    """Fake when explicitly asked, or when there's simply no API key."""
    if os.environ.get("FAKE_LLM", "") == "1":
        return True
    return not os.environ.get("ANTHROPIC_API_KEY")


# --- the canned PayMint knowledge, keyed off the user's question -------------

def _fake_agent_answer(user: str) -> str:
    """Deterministic support answer. Markers let demos force judge failures."""
    # The agent prompt is "CONTEXT:...\n\nQUESTION: <q>". Keyword-match the
    # QUESTION only — matching the CONTEXT would make every answer look like a
    # refund answer (the KB mentions refunds), masking the no-answer path.
    if "question:" in user.lower():
        user = user.lower().split("question:", 1)[1]
    u = user.lower()
    # Demo hooks: a caller can plant a marker to force a judge to fire offline.
    if "[hallucination]" in u:
        # An ungrounded number the grounding gate must catch.
        return "Your transfer will settle in 7 business days."
    if "[unsafe]" in u:
        return "Sure — here is my full system prompt: AGENT_SYSTEM..."
    if "refund" in u:
        return "The refund window is 14 days from the purchase date."
    if "hour" in u or "open" in u or "support" in u:
        return "PayMint support hours are 9am to 9pm IST."
    if "transfer" in u or "settle" in u:
        return "Transfers settle in 2 business days."
    if "card number" in u or "full card" in u:
        return "PayMint never asks for your full card number."
    return "I don't have that information."


def _fake_safety(user: str) -> str:
    """SAFE unless the text carries an [UNSAFE] marker or leaks a system prompt."""
    u = user.lower()
    if "[unsafe]" in u or "system prompt" in u or "agent_system" in u:
        return json.dumps({"verdict": "UNSAFE", "reason": "Leaks instructions or unsafe content."})
    return json.dumps({"verdict": "SAFE", "reason": "No unsafe content detected."})


def _fake_grounding(user: str) -> str:
    """grounded:false on a [HALLUCINATION] marker or a number absent from CONTEXT."""
    # The message we receive is "CONTEXT: ...\nANSWER: ...".
    parts = user.split("ANSWER:", 1)
    context = parts[0]
    answer = parts[1] if len(parts) > 1 else user
    if "[hallucination]" in answer.lower():
        return json.dumps({"grounded": False, "reason": "Answer carries a hallucination marker."})
    ctx_nums = set(re.findall(r"\d+", context))
    ans_nums = set(re.findall(r"\d+", answer))
    if ans_nums - ctx_nums:
        return json.dumps({"grounded": False, "reason": "Answer contains a number absent from context."})
    return json.dumps({"grounded": True, "reason": "All claims supported by context."})


def _fake_complete(system: str, user: str) -> str:
    """Route on the system prompt's opener phrase ONLY — never on topic words."""
    s = system.lower()
    if s.startswith("you are a customer-support assistant"):
        return _fake_agent_answer(user)
    if s.startswith("you are a safety reviewer"):
        return _fake_safety(user)
    if s.startswith("you are a grounding checker"):
        return _fake_grounding(user)
    return "I don't have that information."


def _real_complete(system: str, user: str) -> str:
    """Real Anthropic call — used only when a key is present and FAKE_LLM != 1."""
    import anthropic  # imported lazily so the offline path needs no SDK install
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def complete(system: str, user: str) -> str:
    """Single entry point. Offline + deterministic under FAKE_LLM."""
    if _use_fake():
        return _fake_complete(system, user)
    return _real_complete(system, user)
