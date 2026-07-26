"""
Session 32 — llm_client.py

A deliberately thin Anthropic wrapper (mirrors S31's client).

What this file exposes:
  - LLMClient.complete(system, user, max_tokens) -> str
      One-shot text completion. Used by summariser.py and agent.py.

  - LLMClient.complete_turn(system, messages, max_tokens) -> str
      Multi-message turn used by agent.py when the full context has
      already been assembled by ContextBuilder.

The fake-mode path returns deterministic offline replies keyed by simple
substring matches against the prompt. This lets demo.py run without an
API key — useful for classrooms with flaky internet and for the
Session_32 smoke tests in CI.

To toggle:
    real:  python demo.py 1        (default — needs ANTHROPIC_API_KEY)
    fake:  FAKE_LLM=1 python demo.py 1
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


class LLMClient:
    """
    Thin wrapper around Anthropic's Python SDK.

    Two public methods, no retries, no streaming, no caching. The S26
    observability and S30 action-log patterns belong OUTSIDE this client
    — the client should remain the simplest thing that calls the API.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        self.model = model
        self._fake = os.environ.get("FAKE_LLM", "") == "1"
        self._client = None

    # ------------------------------------------------------------------
    # complete — single user message
    # ------------------------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 600,
    ) -> str:
        if self._fake:
            return _fake_complete(system, user)

        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts: List[str] = []
        for block in message.content:
            if getattr(block, "type", "") == "text":
                parts.append(block.text)
        return "".join(parts).strip()

    # ------------------------------------------------------------------
    # complete_turn — full pre-assembled message list
    # ------------------------------------------------------------------

    def complete_turn(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 600,
    ) -> str:
        if self._fake:
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = str(m.get("content", ""))
                    break
            return _fake_complete(system, last_user, messages=messages)

        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        parts: List[str] = []
        for block in message.content:
            if getattr(block, "type", "") == "text":
                parts.append(block.text)
        return "".join(parts).strip()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic SDK not installed. Run 'pip install anthropic' "
                    "or set FAKE_LLM=1 to use the offline path."
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client


# ----------------------------------------------------------------------------
# Offline fake responses — keyed by substring matches on the user prompt
# ----------------------------------------------------------------------------


def _fake_complete(
    system: str,
    user: str,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Deterministic offline replies for the Session 32 demos.

    The canned set is intentionally narrow. Each demo in demo.py hits a
    specific branch. Adding more demos? Add a branch here and keep the
    rest untouched.
    """

    sys_l = system.lower()
    usr_l = user.lower()

    # ----- Summariser -----
    if "summariser inside an agent's memory system" in sys_l:
        lines: List[str] = []
        capture = False
        for raw in user.splitlines():
            line = raw.strip()
            if line.startswith("NEW_TURNS"):
                capture = True
                continue
            if not line:
                continue
            if capture and line.startswith("- ["):
                lines.append(line[3:])
        if not lines:
            prior = _extract_prior_summary(user)
            return prior or ""
        digest = "; ".join(_compress(l) for l in lines[:6])
        prior = _extract_prior_summary(user)
        if prior and prior != "(empty)":
            return f"{prior.rstrip('.')} New: {digest}."
        return f"Earlier in the conversation: {digest}."

    # ----- Agent (main reasoner) -----
    if "agentic ai memory demo" in sys_l or "you are a helpful assistant" in sys_l:
        haystack = user
        if messages:
            haystack = "\n".join(
                str(m.get("content", "")) for m in messages
            ) + "\n" + user
        haystack_l = haystack.lower()

        if "what is my name" in usr_l or "remind me my name" in usr_l:
            name = _find_word_after(haystack_l, "my name is ")
            if name:
                return f"Your name is {name.title()}."
            return "I don't have your name in memory yet — could you tell me?"
        if "favourite city" in usr_l or "favorite city" in usr_l:
            city = _find_word_after(haystack_l, "favourite city is ")
            if not city:
                city = _find_word_after(haystack_l, "favorite city is ")
            if city:
                return f"You said your favourite city is {city.title()}."
            return "I don't have your favourite city in memory yet."
        if "project" in usr_l and ("name" in usr_l or "called" in usr_l):
            proj = _find_word_after(haystack_l, "project is called ")
            if not proj:
                proj = _find_word_after(haystack_l, "building prepdeck")
                if proj is not None and proj == "":
                    proj = "PrepDeck"
            if proj:
                return f"Your project is called {proj}."
            return "I don't see a project name in memory."
        if "month" in usr_l and "launch" in usr_l:
            month = _find_after(haystack_l, "target launch month is ")
            if not month:
                month = _find_after(haystack_l, "target launch is ")
            if not month:
                month = _find_after(haystack_l, "launch month is ")
            if month:
                return f"You said your launch month is {month.title()}."
            return "I don't have your launch month in memory."
        if "fact" in usr_l and (
            "first" in usr_l or "earlier" in usr_l or "told you" in usr_l
        ):
            fact = _find_after(haystack_l, "fact one: ")
            if fact:
                return f"Earlier you told me — Fact One: {fact}."
            return "I don't have that earlier fact in memory."

        return f"(fake reply) noted: {user.strip()[:140]}"

    return ""


def _extract_prior_summary(prompt: str) -> str:
    """Pull the PRIOR_SUMMARY block out of the summariser user prompt."""

    capturing = False
    captured: List[str] = []
    for line in prompt.splitlines():
        if line.startswith("PRIOR_SUMMARY:"):
            capturing = True
            continue
        if line.startswith("NEW_TURNS"):
            break
        if capturing:
            captured.append(line)
    return "\n".join(captured).strip()


def _compress(line: str) -> str:
    """Strip the role tag and collapse repeated whitespace."""

    if "] " in line:
        line = line.split("] ", 1)[1]
    return " ".join(line.split())[:80]


def _find_after(haystack: str, needle: str) -> str:
    """Return text following `needle`, stopping at sentence boundaries."""

    idx = haystack.find(needle)
    if idx < 0:
        return ""
    tail = haystack[idx + len(needle) :]
    stop_chars = ".!?\n,;\"'<>{}[]"
    out_chars: List[str] = []
    for ch in tail:
        if ch in stop_chars:
            break
        out_chars.append(ch)
        if len(out_chars) >= 60:
            break
    return "".join(out_chars).strip()


def _find_word_after(haystack: str, needle: str) -> str:
    """
    Like _find_after but also stops at conjunction words.

    Used for the favourite-city router so an input like
    "my favourite city is Lisbon and my target..." returns just "lisbon"
    rather than the whole tail.
    """

    raw = _find_after(haystack, needle)
    if not raw:
        return ""
    stop_words = (" and ", " but ", " however ", " though ", " or ")
    earliest = len(raw)
    for sw in stop_words:
        i = raw.find(sw)
        if 0 <= i < earliest:
            earliest = i
    return raw[:earliest].strip()
