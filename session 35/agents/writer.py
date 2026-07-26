"""
Session 35 — agents/writer.py

Leaf worker that composes prose from a fact list. Same role boundary
as S34: writes only, never researches, never verifies.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List

from agent_types import Fact
from llm_client import LLMClient
from prompts import WRITER_SYSTEM


class Writer:
    def __init__(self, llm: LLMClient,
                 trace: Callable[[Dict], None]) -> None:
        self.llm = llm
        self.trace = trace

    def run(self, facts: List[Fact], revision_instruction: str = "") -> str:
        facts_json = json.dumps([
            {"text": f.text, "source": f.source} for f in facts
        ])
        user_parts = [
            "USER_REQUEST: Compose a 3-paragraph brief.",
            f"FACTS:\n{facts_json}",
        ]
        if revision_instruction:
            user_parts.append(f"REVISE: {revision_instruction}")
            user_parts.append("issues=[\"address fact-checker feedback\"]")
        user = "\n".join(user_parts)
        draft = self.llm.complete(WRITER_SYSTEM, user)
        words = len(draft.split())
        self.trace({
            "type": "writer_finished",
            "revision": bool(revision_instruction),
            "words": words,
        })
        return draft
