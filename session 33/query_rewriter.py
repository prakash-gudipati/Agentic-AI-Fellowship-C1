"""
Session 33 — query_rewriter.py

Two helpers an agent uses BEFORE it calls search_kb:

  rewrite(raw_question) -> str
      Tighten a chatty user question into a retrieval-friendly query.

  decompose(raw_question) -> List[str]
      Split a compound question ("X and Y") into independent sub-queries.

Why these are separate from the agent loop:
  - You can imagine a future where the agent does its own rewriting
    inside the system prompt. In practice, naming rewrite/decompose as
    explicit functions keeps the trace logger honest — every transform
    on the user's question shows up as its own line.
  - Decomposition is a SHORT-CIRCUIT. If the question decomposes into
    two sub-questions, the agent retrieves twice without burning a
    full decision turn each time. That's the Decompose-then-Retrieve
    pattern.
"""

from __future__ import annotations

import json
from typing import List, Optional

from llm_client import LLMClient
from prompts import DECOMPOSER_SYSTEM, REWRITER_SYSTEM


def rewrite(raw_question: str, llm: Optional[LLMClient] = None) -> str:
    """Return a tightened retrieval query.

    Cheap (one LLM call). Always succeeds — if the LLM returns
    something weird, fall back to the raw question.
    """

    llm = llm or LLMClient()
    user_prompt = (
        f"USER_QUESTION:\n{raw_question}\n\n"
        "Rewrite this as a retrieval query. Output the query only."
    )
    out = llm.complete(REWRITER_SYSTEM, user_prompt, max_tokens=80)
    out = out.strip().strip('"').strip("'")
    if not out:
        return raw_question.strip()
    # Defensive: cap at 200 chars so a chatty model doesn't poison the
    # vector search.
    return out[:200]


def decompose(raw_question: str, llm: Optional[LLMClient] = None) -> List[str]:
    """Split a compound question into atomic sub-questions.

    Returns a list of strings. A single-element list means the
    question was already atomic.
    """

    llm = llm or LLMClient()
    user_prompt = (
        f"USER_QUESTION:\n{raw_question}\n\n"
        "If this is a compound question, decompose it. Output a JSON "
        "array of strings. If it's already atomic, output a JSON array "
        "with the original question as the only element."
    )
    out = llm.complete(DECOMPOSER_SYSTEM, user_prompt, max_tokens=200)
    out = out.strip()

    parsed: Optional[List[str]] = None
    try:
        candidate = json.loads(out)
        if isinstance(candidate, list) and all(
            isinstance(x, str) for x in candidate
        ):
            parsed = [x.strip() for x in candidate if x.strip()]
    except json.JSONDecodeError:
        parsed = None

    if not parsed:
        # Heuristic fallback: split on " and " or " plus ".
        for splitter in (" and ", " plus "):
            if splitter in raw_question.lower():
                parts = [
                    p.strip(" ?.") for p in raw_question.split(splitter, 1)
                ]
                if len(parts) == 2 and all(parts):
                    return parts
        return [raw_question.strip()]

    return parsed
