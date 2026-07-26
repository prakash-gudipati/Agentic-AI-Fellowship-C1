"""
Session 35 — agents/researcher.py

Leaf worker that gathers facts on ONE sub-topic. Mirrors the S34
researcher but takes a sub-topic from a Research Manager rather than
directly from an Orchestrator.

Role boundary (from prompts.RESEARCHER_SYSTEM):
  - gather facts ONLY
  - never write paragraphs
  - never split topics
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List

from agent_types import Fact
from llm_client import LLMClient
from prompts import RESEARCHER_SYSTEM


class Researcher:
    def __init__(self, llm: LLMClient, researcher_id: str,
                 trace: Callable[[Dict], None]) -> None:
        self.llm = llm
        self.researcher_id = researcher_id
        self.trace = trace

    def run(self, sub_topic: str) -> List[Fact]:
        user = (
            f"USER_REQUEST: Gather facts on the sub-topic.\n"
            f"sub-topic on: {sub_topic}\n"
        )
        raw = self.llm.complete(RESEARCHER_SYSTEM, user)
        facts = _parse_fact_array(raw)
        self.trace({
            "type": "researcher_finished",
            "researcher_id": self.researcher_id,
            "num_facts": len(facts),
        })
        return facts


def _parse_fact_array(raw: str) -> List[Fact]:
    raw = raw.strip()
    # Strip ```json fences if present.
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    facts: List[Fact] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        source = str(item.get("source", "")).strip()
        if text and source:
            facts.append(Fact(text=text, source=source))
    return facts
