"""
Session 34 — agents/researcher.py

The researcher worker. Gathers facts on a topic. Writes the result to
the FACTS section of the shared scratchpad. PROD PATTERN: Role
Specialization — this agent has ONE job and ONE output shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from agent_types import Fact
from llm_client import LLMClient
from prompts import RESEARCHER_SYSTEM
from scratchpad import SECTION_FACTS, Scratchpad


@dataclass
class Researcher:
    llm: Optional[LLMClient] = None

    def __post_init__(self) -> None:
        self.llm = self.llm or LLMClient()
        self.name = "researcher"

    def run(
        self, instruction: str, scratchpad: Scratchpad, round_no: int
    ) -> List[Fact]:
        user_prompt = (
            f"INSTRUCTION:\n{instruction}\n\n"
            "Return the JSON fact array specified in your system prompt."
        )
        raw = self.llm.complete(RESEARCHER_SYSTEM, user_prompt, max_tokens=600)
        facts = _parse_facts(raw)
        scratchpad.write(
            SECTION_FACTS,
            facts,
            agent=self.name,
            round_no=round_no,
            summary=f"{len(facts)} facts gathered",
        )
        return facts


def _parse_facts(raw: str) -> List[Fact]:
    """Parse the researcher's JSON output. Defensive against extra prose."""

    raw = raw.strip()
    # Try to find the first JSON array in the response.
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end < 0 or end < start:
        return []
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    out: List[Fact] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "text" in item:
                out.append(
                    Fact(
                        text=str(item["text"]).strip(),
                        source=str(item.get("source", "")).strip(),
                    )
                )
    return out
