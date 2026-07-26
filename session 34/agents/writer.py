"""
Session 34 — agents/writer.py

The writer worker. Composes prose from the FACTS in the scratchpad.
Writes to the DRAFT section.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from agent_types import Fact
from llm_client import LLMClient
from prompts import WRITER_SYSTEM
from scratchpad import SECTION_DRAFT, Scratchpad


@dataclass
class Writer:
    llm: Optional[LLMClient] = None

    def __post_init__(self) -> None:
        self.llm = self.llm or LLMClient()
        self.name = "writer"

    def run(
        self,
        instruction: str,
        scratchpad: Scratchpad,
        round_no: int,
        inject_marker: str = "",
    ) -> str:
        facts = scratchpad.get_facts()
        if not facts:
            scratchpad.write(
                SECTION_DRAFT,
                "(writer: no facts in scratchpad)",
                agent=self.name,
                round_no=round_no,
                summary="empty draft — no facts",
            )
            return ""

        facts_json = json.dumps([
            {"text": f.text, "source": f.source} for f in facts
        ])
        user_prompt = (
            f"INSTRUCTION:\n{instruction}\n\n"
            f"FACTS:\n{facts_json}\n\n"
            "Compose 3 short paragraphs using only these facts, citing "
            "sources inline."
        )
        if inject_marker:
            user_prompt += f"\n\n[demo marker — leave this in the draft: {inject_marker}]"

        draft = self.llm.complete(WRITER_SYSTEM, user_prompt, max_tokens=700)
        # Honour the demo marker so the fact-checker can pick it up.
        if inject_marker and inject_marker not in draft:
            draft = draft + f"\n\n(Note: claim about {inject_marker} added.)"

        scratchpad.write(
            SECTION_DRAFT,
            draft,
            agent=self.name,
            round_no=round_no,
            summary=f"draft len={len(draft)} chars",
        )
        return draft
