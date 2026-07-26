"""
Session 38 — llm.py

The model layer. ONE real model object, shared by every node.

This session uses a real LLM end to end — there is no offline fake mode. The
instructor runs it live with an ANTHROPIC_API_KEY in the environment. We wrap
the raw ChatAnthropic object in three tiny, single-purpose functions so the
nodes in nodes.py read like prose, not like API plumbing.

[NEW TERM: ChatAnthropic] — LangChain's adapter around Anthropic's Claude
models. It is a Runnable, so it snaps into any LCEL pipe just like in S37.
"""

from __future__ import annotations

import json
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

import prompts

# Claude Haiku 4.5 — cheap and fast, which matters when a loop may call the
# model several times per run. Override with S38_MODEL if you want.
_MODEL_NAME = os.environ.get("S38_MODEL", "claude-haiku-4-5-20251001")

# Built once and reused. Building a fresh client per call is a common beginner
# cost/latency bug.
_MODEL: ChatAnthropic | None = None


def get_chat_model() -> ChatAnthropic:
    """Return the one shared ChatAnthropic instance (built on first use)."""
    global _MODEL
    if _MODEL is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. This session runs against the "
                "real Anthropic API. Copy .env.example to .env and add your key, "
                "or run `python demo.py --selftest` to test the graph wiring "
                "without any API calls."
            )
        _MODEL = ChatAnthropic(model=_MODEL_NAME, temperature=0, max_tokens=1024)
    return _MODEL


def _ask(system_prompt: str, user_text: str) -> str:
    """Single-turn call: one system prompt, one user message, text back."""
    model = get_chat_model()
    reply = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ])
    return reply.content.strip() if isinstance(reply.content, str) else str(reply.content)


def analyze_query(question: str) -> str:
    """Turn a research question into one focused web-search query."""
    return _ask(prompts.ANALYZER_SYSTEM, question).strip().strip('"')


def evaluate_results(question: str, results: list) -> dict:
    """The quality gate. Score how well `results` answer `question`.

    Returns {"score": float, "reason": str, "refined_query": str}. We parse
    the model's JSON defensively: a gate that crashes on a stray character is
    worse than a gate that degrades to a cautious low score.
    """
    snippets = _format_snippets(results)
    raw = _ask(
        prompts.EVALUATOR_SYSTEM,
        f"QUESTION:\n{question}\n\nSEARCH RESULTS:\n{snippets}",
    )
    parsed = _parse_json_object(raw)
    score = _clamp_float(parsed.get("score"), default=0.0)
    reason = str(parsed.get("reason") or "no reason given")
    refined = str(parsed.get("refined_query") or "").strip() or question
    return {"score": score, "reason": reason, "refined_query": refined}


def write_report(question: str, results: list) -> str:
    """Write the final grounded report from the surviving search results."""
    snippets = _format_snippets(results)
    return _ask(
        prompts.REPORTER_SYSTEM,
        f"QUESTION:\n{question}\n\nSEARCH RESULTS:\n{snippets}",
    )


# --------------------------------------------------------------------------
# small helpers — parsing and formatting
# --------------------------------------------------------------------------


def _format_snippets(results: list) -> str:
    """Render search hits as a compact numbered block for the prompt."""
    if not results:
        return "(no results)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', '')}\n    {r.get('content', '')}\n    URL: {r.get('url', '')}")
    return "\n".join(lines)


def _parse_json_object(text: str) -> dict:
    """Best-effort: pull the first {...} block and parse it. Never raises."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {}


def _clamp_float(value, default: float) -> float:
    """Coerce to a 0.0-1.0 float; fall back to `default` on junk."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
