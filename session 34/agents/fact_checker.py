"""
Session 34 — agents/fact_checker.py

The fact-checker worker. Reads the DRAFT and FACTS sections, emits a
Critique with verdict ACCEPT or REVISE. Writes the verdict to the
CRITIQUE section.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from agent_types import Critique, Fact
from llm_client import LLMClient
from prompts import FACT_CHECKER_SYSTEM
from scratchpad import SECTION_CRITIQUE, Scratchpad


@dataclass
class FactChecker:
    llm: Optional[LLMClient] = None

    def __post_init__(self) -> None:
        self.llm = self.llm or LLMClient()
        self.name = "fact_checker"

    def run(
        self, instruction: str, scratchpad: Scratchpad, round_no: int
    ) -> Critique:
        draft = scratchpad.get_draft()
        facts = scratchpad.get_facts()

        facts_lines = "\n".join(
            f"  - text=\"{f.text}\" source=\"{f.source}\"" for f in facts
        )
        user_prompt = (
            f"INSTRUCTION:\n{instruction}\n\n"
            f"DRAFT:\n{draft}\n\n"
            f"FACTS:\n{facts_lines}\n\n"
            "Output JSON with verdict, issues, notes — as your system "
            "prompt specifies."
        )

        raw = self.llm.complete(FACT_CHECKER_SYSTEM, user_prompt, max_tokens=400)
        critique = _parse_critique(raw)
        scratchpad.write(
            SECTION_CRITIQUE,
            critique,
            agent=self.name,
            round_no=round_no,
            summary=f"verdict={critique.verdict} issues={len(critique.issues)}",
        )
        return critique


def _parse_critique(raw: str) -> Critique:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0:
        return Critique(verdict="REVISE", issues=["unparseable output"], notes=raw[:200])
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return Critique(verdict="REVISE", issues=["JSON parse error"], notes=raw[:200])
    verdict = str(parsed.get("verdict", "REVISE")).upper()
    if verdict not in ("ACCEPT", "REVISE"):
        verdict = "REVISE"
    issues = parsed.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    return Critique(
        verdict=verdict,
        issues=[str(i) for i in issues],
        notes=str(parsed.get("notes", "")),
    )
