"""
Session 35 — agents/fact_checker.py

Reviews a draft against a fact list. Emits ACCEPT or REVISE. Same role
boundary as S34.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Tuple

from agent_types import Fact
from llm_client import LLMClient
from prompts import FACT_CHECKER_SYSTEM


class FactChecker:
    def __init__(self, llm: LLMClient,
                 trace: Callable[[Dict], None]) -> None:
        self.llm = llm
        self.trace = trace

    def run(self, draft: str, facts: List[Fact]) -> Tuple[str, List[str]]:
        facts_block = "\n".join(
            f"- text=\"{f.text}\" source=\"{f.source}\"" for f in facts
        )
        user = (
            "USER_REQUEST: Verify the draft against the facts.\n"
            f"DRAFT:\n{draft}\n\n"
            f"FACT_LIST:\n{facts_block}"
        )
        raw = self.llm.complete(FACT_CHECKER_SYSTEM, user)
        verdict, issues = _parse_verdict(raw)
        self.trace({
            "type": "fact_check_verdict",
            "verdict": verdict,
            "issues": issues,
        })
        return verdict, issues


def _parse_verdict(raw: str) -> Tuple[str, List[str]]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return "REVISE", ["unparseable verdict"]
    if not isinstance(parsed, dict):
        return "REVISE", ["non-dict verdict"]
    verdict = str(parsed.get("verdict", "REVISE")).upper()
    if verdict not in {"ACCEPT", "REVISE"}:
        verdict = "REVISE"
    issues_raw = parsed.get("issues", [])
    issues = [str(x) for x in issues_raw] if isinstance(issues_raw, list) else []
    return verdict, issues
