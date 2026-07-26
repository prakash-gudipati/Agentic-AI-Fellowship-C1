"""
Session 35 — agents/competitive_panel.py

PROD PATTERN: Best-of-N Judging.

Run N candidate writers in parallel, each constrained to a different
STYLE. A Judge agent scores all N drafts on three criteria and picks
a single winner with rationale.

CONTRAST with Debate (debate_panel.py):
  - Debate produces a consensus.
  - Competitive picks a winner.

Use this pattern when output quality varies substantially across
attempts and you want explicit, auditable scoring rather than the
first-good-enough answer.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

from agent_types import (
    CandidateDraft, CompetitiveResult, Fact, JudgeVerdict,
)
from llm_client import LLMClient
from messages import MessageBus
from prompts import CANDIDATE_WRITER_SYSTEM, JUDGE_SYSTEM


DEFAULT_STYLES = ["concise", "detailed", "narrative"]


class CompetitivePanel:
    def __init__(self, llm: LLMClient,
                 trace: Callable[[Dict], None],
                 styles: Optional[List[str]] = None) -> None:
        self.llm = llm
        self.trace = trace
        self.styles = styles or DEFAULT_STYLES
        self.bus = MessageBus()

    # ------------------------------------------------------------------
    # Public entry point.
    # ------------------------------------------------------------------

    def run(self, topic: str, facts: List[Fact]) -> CompetitiveResult:
        self.llm.reset_count()
        candidates = self._run_candidates(topic, facts)
        verdict = self._judge(candidates, facts)
        return CompetitiveResult(
            verdict=verdict,
            candidates=candidates,
            total_llm_calls=self.llm.call_count,
        )

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _run_candidates(self, topic: str,
                        facts: List[Fact]) -> List[CandidateDraft]:
        facts_json = json.dumps([
            {"text": f.text, "source": f.source} for f in facts
        ])
        plans: List[tuple] = []
        for i, style in enumerate(self.styles):
            cid = f"candidate_{i + 1}"
            user = (
                f"USER_REQUEST: Write a brief.\n"
                f"TOPIC: {topic}\n"
                f"STYLE: {style}\n"
                f"FACTS:\n{facts_json}\n"
            )
            plans.append((cid, style, user))

        candidates: List[CandidateDraft] = []
        with ThreadPoolExecutor(max_workers=len(plans)) as ex:
            futures = []
            for cid, style, user in plans:
                fut = ex.submit(self.llm.complete, CANDIDATE_WRITER_SYSTEM, user)
                futures.append((cid, style, fut))
            for cid, style, fut in futures:
                draft = fut.result()
                candidate = CandidateDraft(
                    candidate_id=cid, draft=draft, style=style,
                )
                candidates.append(candidate)
                self.trace({
                    "type": "candidate_drafted",
                    "candidate_id": cid,
                    "style": style,
                    "words": len(draft.split()),
                })
                self.bus.report(
                    sender=cid, recipient="judge",
                    subject=f"candidate brief style={style}",
                    payload={"draft": draft, "style": style},
                    round_num=0,
                )
                self.trace({
                    "type": "message_sent",
                    "sender": cid, "recipient": "judge",
                    "intent": "REPORT",
                    "subject": f"candidate brief style={style}",
                })
        return candidates

    def _judge(self, candidates: List[CandidateDraft],
               facts: List[Fact]) -> JudgeVerdict:
        facts_block = "\n".join(
            f"- text=\"{f.text}\" source=\"{f.source}\"" for f in facts
        )
        candidates_block = "\n\n".join(
            f"[{c.candidate_id}] style={c.style}\nDRAFT:\n{c.draft}"
            for c in candidates
        )
        user = (
            "USER_REQUEST: Score and pick a winner.\n"
            f"FACT_LIST:\n{facts_block}\n\n"
            f"CANDIDATES:\n{candidates_block}\n"
        )
        raw = self.llm.complete(JUDGE_SYSTEM, user)
        verdict = _parse_verdict(raw)
        self.trace({
            "type": "judge_verdict",
            "winner_id": verdict.winner_id,
            "scores": verdict.scores,
        })
        return verdict


def _parse_verdict(raw: str) -> JudgeVerdict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return JudgeVerdict(
            winner_id="(unparseable)", rationale=raw, scores={},
        )
    if not isinstance(parsed, dict):
        return JudgeVerdict(
            winner_id="(non-dict)", rationale=str(parsed), scores={},
        )
    return JudgeVerdict(
        winner_id=str(parsed.get("winner_id", "")),
        rationale=str(parsed.get("rationale", "")),
        scores=parsed.get("scores", {}) if isinstance(parsed.get("scores"), dict) else {},
    )
