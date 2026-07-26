"""
Session 35 — agents/debate_panel.py

PROD PATTERN: Consensus Synthesis (the headline of Pattern 2 — Debate).

A debate panel runs three panelists in parallel — bull, bear, neutral —
each arguing a different position on the same topic. A moderator then
synthesises their arguments into a ConsensusReport.

CONTRAST with S34's Critique Loop:
  - Critique Loop is generator + reviewer; output is one accepted
    artifact.
  - Debate is multiple peers + synthesiser; output is a structured
    consensus that respects all viewpoints.

CONTRAST with the Competitive panel (next file):
  - Debate produces a CONSENSUS that respects all sides.
  - Competitive PICKS A WINNER (best-of-N).

Role boundaries are encoded in the panelist prompts. Each prompt
forbids the OTHER panelists' jobs, which is what makes the debate
substantive instead of three echoes.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List

from agent_types import Argument, ConsensusReport, DebateResult, Message
from llm_client import LLMClient
from messages import MessageBus
from prompts import (
    BULL_PANELIST_SYSTEM,
    BEAR_PANELIST_SYSTEM,
    NEUTRAL_PANELIST_SYSTEM,
    MODERATOR_SYSTEM,
)


class DebatePanel:
    def __init__(self, llm: LLMClient,
                 trace: Callable[[Dict], None],
                 rounds: int = 1) -> None:
        self.llm = llm
        self.trace = trace
        self.rounds = max(1, rounds)
        self.bus = MessageBus()

    # ------------------------------------------------------------------
    # Public entry point.
    # ------------------------------------------------------------------

    def run(self, topic: str) -> DebateResult:
        self.llm.reset_count()
        transcript: List[Argument] = []

        for round_num in range(self.rounds):
            self.trace({"type": "debate_round_start", "round_num": round_num + 1})
            args = self._run_panel_round(topic, round_num)
            transcript.extend(args)

        # Step 2 — moderator synthesises.
        consensus = self._moderate(topic, transcript)
        return DebateResult(
            consensus=consensus,
            transcript=transcript,
            total_llm_calls=self.llm.call_count,
            rounds_used=self.rounds,
        )

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _run_panel_round(self, topic: str, round_num: int) -> List[Argument]:
        panelists = [
            ("bull_panelist", BULL_PANELIST_SYSTEM),
            ("bear_panelist", BEAR_PANELIST_SYSTEM),
            ("neutral_panelist", NEUTRAL_PANELIST_SYSTEM),
        ]
        # Fan out in parallel — all three have the same input shape.
        with ThreadPoolExecutor(max_workers=len(panelists)) as ex:
            futures = []
            for name, prompt in panelists:
                user = (
                    f"USER_REQUEST: Argue your position.\n"
                    f"TOPIC: {topic}\n"
                    f"ROUND: {round_num + 1}\n"
                )
                futures.append((name, ex.submit(self.llm.complete, prompt, user)))
            args: List[Argument] = []
            for name, fut in futures:
                raw = fut.result()
                arg = _parse_argument(name, raw, round_num + 1)
                if arg:
                    args.append(arg)
                    self.trace({
                        "type": "panelist_argument",
                        "panelist": name,
                        "claim": arg.claim,
                    })
                    self.bus.report(
                        sender=name, recipient="moderator",
                        subject=f"round {round_num + 1} argument",
                        payload={"claim": arg.claim, "evidence": arg.evidence},
                        round_num=round_num,
                    )
                    self.trace({
                        "type": "message_sent",
                        "sender": name, "recipient": "moderator",
                        "intent": "REPORT",
                        "subject": f"round {round_num + 1} argument",
                    })
        return args

    def _moderate(self, topic: str,
                  transcript: List[Argument]) -> ConsensusReport:
        # Compose a moderator prompt containing the full transcript.
        lines = []
        for arg in transcript:
            lines.append(
                f"[{arg.panelist} round {arg.round_num}] claim={arg.claim!r} "
                f"evidence={arg.evidence}"
            )
        user = (
            f"USER_REQUEST: Synthesise the panel.\n"
            f"TOPIC: {topic}\n"
            f"TRANSCRIPT:\n" + "\n".join(lines) + "\n"
        )
        raw = self.llm.complete(MODERATOR_SYSTEM, user)
        report = _parse_consensus(raw)
        self.trace({"type": "consensus", "report": {
            "agreed_points": report.agreed_points,
            "disagreements": report.disagreements,
            "confidence": report.confidence,
        }})
        return report


def _parse_argument(panelist: str, raw: str, round_num: int):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    claim = str(parsed.get("claim", "")).strip()
    if not claim:
        return None
    evidence = parsed.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    return Argument(
        panelist=panelist,
        claim=claim,
        evidence=[str(x) for x in evidence],
        round_num=round_num,
    )


def _parse_consensus(raw: str) -> ConsensusReport:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ConsensusReport(
            agreed_points=[],
            disagreements=[],
            final_position=raw or "(no consensus parsed)",
            confidence=0.0,
        )
    if not isinstance(parsed, dict):
        return ConsensusReport(
            agreed_points=[], disagreements=[],
            final_position=str(parsed), confidence=0.0,
        )
    agreed = parsed.get("agreed_points", [])
    disagreements = parsed.get("disagreements", [])
    final_position = parsed.get("final_position", "")
    confidence = parsed.get("confidence", 0.0)
    try:
        confidence_f = float(confidence)
    except (TypeError, ValueError):
        confidence_f = 0.0
    return ConsensusReport(
        agreed_points=[str(x) for x in agreed] if isinstance(agreed, list) else [],
        disagreements=[str(x) for x in disagreements] if isinstance(disagreements, list) else [],
        final_position=str(final_position),
        confidence=confidence_f,
    )
