"""
Session 39 — llm.py

The model layer. ONE real model object, shared by every node — plus an OFFLINE
FAKE mode so every demo, the selftest, and the instructor's pre-session check
can run with no API key and no network.

Why a fake mode this session (S38 had none)? Because S39's headline ideas —
checkpointing, human-in-the-loop, the Store, time-travel — are all FRAMEWORK
wiring. None of them depend on what the model actually says. The fake brain
lets us prove the wiring works offline; the real brain is swapped in live by
the instructor with ANTHROPIC_API_KEY set.

  Real mode (default when a key is present): ChatAnthropic, Claude Haiku 4.5.
  Fake mode (FAKE_LLM=1, or no key + offline):
      canned replies routed on the system prompt's opener phrase.

[NEW TERM: ChatAnthropic] — LangChain's adapter around Claude. A Runnable, so
it snaps into any LCEL pipe exactly like in S37.
"""

from __future__ import annotations

import json
import os

import prompts

_MODEL_NAME = os.environ.get("S39_MODEL", "claude-haiku-4-5-20251001")
_MODEL = None  # built once, on first real use


def _fake_enabled() -> bool:
    """Fake mode is ON if FAKE_LLM is truthy, or if no API key is available."""
    if os.environ.get("FAKE_LLM", "").lower() in ("1", "true", "yes"):
        return True
    return not os.environ.get("ANTHROPIC_API_KEY")


def get_chat_model():
    """Return the one shared ChatAnthropic instance (built on first use)."""
    global _MODEL
    if _MODEL is None:
        from langchain_anthropic import ChatAnthropic  # imported lazily
        _MODEL = ChatAnthropic(model=_MODEL_NAME, temperature=0, max_tokens=1024)
    return _MODEL


def _ask(system_prompt: str, user_text: str) -> str:
    """Single-turn call: one system prompt, one user message, text back.

    Routes to the fake brain when fake mode is enabled.
    """
    if _fake_enabled():
        return _fake_reply(system_prompt, user_text)

    from langchain_core.messages import HumanMessage, SystemMessage
    model = get_chat_model()
    reply = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ])
    return reply.content.strip() if isinstance(reply.content, str) else str(reply.content)


# --------------------------------------------------------------------------
# the three (now four) thinking steps the nodes call
# --------------------------------------------------------------------------


def make_plan(question: str, prior_context: str = "") -> str:
    """NEW in S39. Write a short, human-reviewable research plan."""
    user = question if not prior_context else f"{question}\n\nWHAT WE ALREADY KNOW:\n{prior_context}"
    return _ask(prompts.PLANNER_SYSTEM, user).strip()


def analyze_query(question: str, approved_plan: str = "") -> str:
    """Turn the approved plan + question into one focused search query."""
    user = question if not approved_plan else f"PLAN:\n{approved_plan}\n\nQUESTION:\n{question}"
    return _ask(prompts.ANALYZER_SYSTEM, user).strip().strip('"')


def evaluate_results(question: str, results: list) -> dict:
    """The quality gate. Score how well `results` answer `question`.

    Parses the model's JSON defensively: a gate that crashes on a stray
    character is worse than one that degrades to a cautious low score.
    """
    snippets = _format_snippets(results)
    raw = _ask(prompts.EVALUATOR_SYSTEM, f"QUESTION:\n{question}\n\nSEARCH RESULTS:\n{snippets}")
    parsed = _parse_json_object(raw)
    return {
        "score": _clamp_float(parsed.get("score"), default=0.0),
        "reason": str(parsed.get("reason") or "no reason given"),
        "refined_query": str(parsed.get("refined_query") or "").strip() or question,
    }


def write_report(question: str, results: list) -> str:
    """Write the final grounded report from the surviving search results."""
    snippets = _format_snippets(results)
    return _ask(prompts.REPORTER_SYSTEM, f"QUESTION:\n{question}\n\nSEARCH RESULTS:\n{snippets}")


# --------------------------------------------------------------------------
# the offline fake brain — routes on the system prompt's opener phrase
# --------------------------------------------------------------------------


def _fake_reply(system_prompt: str, user_text: str) -> str:
    """Canned, deterministic replies for offline runs.

    Each branch matches a DISTINCTIVE opener phrase (Phase 5 rule #2) so routes
    never poach each other. The replies are intentionally plausible so the
    trace reads like a real run.
    """
    sys_l = system_prompt.strip().lower()
    topic = _first_line(user_text)

    if sys_l.startswith("you are a research planner"):
        return (f"1. Define the key terms in: {topic}\n"
                f"2. Find current best practices and named tools\n"
                f"3. Gather one concrete production example\n"
                f"4. Summarise the trade-offs")

    if sys_l.startswith("you are a query analyst"):
        return f"{topic} best practices production guide"

    if sys_l.startswith("you are a relevance evaluator"):
        # Score low on the FIRST attempt (1 result block) so the cycle loops
        # once and the gate is visible; high once results have accumulated.
        num_blocks = user_text.count("URL:") or user_text.lower().count("[1]")
        if num_blocks <= 1:
            return json.dumps({"score": 0.45, "reason": "thin coverage, only one source",
                               "refined_query": f"{topic} detailed examples and trade-offs"})
        return json.dumps({"score": 0.86, "reason": "multiple sources cover the question",
                           "refined_query": topic})

    if sys_l.startswith("you are a report writer"):
        return (f"Based on the gathered sources, here is what we found about "
                f"{topic}: the results converge on a few practical points and "
                f"name the tools teams reach for in production. (Offline fake "
                f"report — run with a real ANTHROPIC_API_KEY for the full answer.)")

    return "(fake llm: no route matched)"


# --------------------------------------------------------------------------
# small helpers — parsing and formatting
# --------------------------------------------------------------------------


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line and not line.endswith(":") and "QUESTION" not in line.upper():
            return line[:80]
    return text.strip()[:80] or "the topic"


def _format_snippets(results: list) -> str:
    if not results:
        return "(no results)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', '')}\n    {r.get('content', '')}\n    URL: {r.get('url', '')}")
    return "\n".join(lines)


def _parse_json_object(text: str) -> dict:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {}


def _clamp_float(value, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
