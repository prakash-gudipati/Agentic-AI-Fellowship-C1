"""
Session 35 — agents/editorial_manager.py

PROD PATTERN: Hierarchical Decomposition (level 1 of 2, sibling of
ResearchManager).

The Editorial Manager runs a small two-agent writer-fact-checker loop
on behalf of the Director. It receives a merged fact list, asks the
writer to draft, asks the fact-checker to review, and either reports
ACCEPTED to the director or escalates after 3 unsuccessful revision
rounds (PROD PATTERN: Termination Conditions from S34 still applies at
every layer).

Role boundary (from prompts.EDITORIAL_MANAGER_SYSTEM):
  - dispatch + loop + report ONLY
  - never write or fact-check
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Tuple

from agent_types import Fact
from llm_client import LLMClient
from messages import MessageBus
from prompts import EDITORIAL_MANAGER_SYSTEM
from agents.writer import Writer
from agents.fact_checker import FactChecker


MAX_REVISION_ROUNDS = 3


class EditorialManager:
    def __init__(self, llm: LLMClient, bus: MessageBus,
                 trace: Callable[[Dict], None]) -> None:
        self.llm = llm
        self.bus = bus
        self.trace = trace
        self.writer = Writer(llm, trace)
        self.fact_checker = FactChecker(llm, trace)

    # ------------------------------------------------------------------
    # End-to-end loop. Returns (final_draft, status).
    # status is one of: "ACCEPTED", "ESCALATED".
    # ------------------------------------------------------------------

    def run(self, facts: List[Fact], round_num: int = 0) -> Tuple[str, str]:
        # Step 1 — initial draft.
        draft = self.writer.run(facts)
        self.bus.report(
            sender="writer", recipient="editorial_manager",
            subject="initial draft",
            payload={"draft": draft}, round_num=round_num,
        )
        self.trace({
            "type": "message_sent",
            "sender": "writer", "recipient": "editorial_manager",
            "intent": "REPORT", "subject": "initial draft",
        })

        # Step 2 — revision loop, bounded by MAX_REVISION_ROUNDS.
        for revision in range(MAX_REVISION_ROUNDS):
            self._action_log(
                action="REVIEW", target="fact_checker",
                instruction="Review the draft against the fact list.",
            )
            verdict, issues = self.fact_checker.run(draft, facts)
            self.bus.review(
                sender="fact_checker", recipient="editorial_manager",
                subject=f"verdict={verdict.lower()}",
                payload={"verdict": verdict, "issues": issues},
                round_num=round_num,
            )
            self.trace({
                "type": "message_sent",
                "sender": "fact_checker", "recipient": "editorial_manager",
                "intent": "REVIEW", "subject": f"verdict={verdict.lower()}",
            })
            if verdict == "ACCEPT":
                self._report_to_director(draft, "ACCEPTED",
                                         round_num=round_num)
                return draft, "ACCEPTED"

            # Verdict REVISE — ask the writer for a revision.
            self._action_log(
                action="REVISE", target="writer",
                instruction=f"Revise. Issues: {'; '.join(issues[:3])}",
            )
            draft = self.writer.run(facts, revision_instruction=
                                    f"Revise to address: {'; '.join(issues[:3])}")
            self.bus.report(
                sender="writer", recipient="editorial_manager",
                subject="revised draft",
                payload={"draft": draft, "revision": revision + 1},
                round_num=round_num,
            )
            self.trace({
                "type": "message_sent",
                "sender": "writer", "recipient": "editorial_manager",
                "intent": "REPORT", "subject": "revised draft",
            })

        # If we fall out of the loop, escalate to the Director.
        self._report_to_director(draft, "ESCALATED", round_num=round_num)
        return draft, "ESCALATED"

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _action_log(self, *, action: str, target: str,
                    instruction: str) -> None:
        self.trace({
            "type": "manager_action",
            "manager": "editorial_manager",
            "action": action,
            "target": target,
            "instruction": instruction,
        })

    def _report_to_director(self, draft: str, status: str,
                            round_num: int = 0) -> None:
        # status is "ACCEPTED" or "ESCALATED"
        intent_subject = (
            "editorial_report=accepted" if status == "ACCEPTED"
            else "editorial_report=escalated"
        )
        self.bus.report(
            sender="editorial_manager", recipient="director",
            subject=intent_subject,
            payload={"draft": draft, "status": status},
            round_num=round_num,
        )
        self.trace({
            "type": "message_sent",
            "sender": "editorial_manager", "recipient": "director",
            "intent": "REPORT", "subject": intent_subject,
        })
        self.trace({
            "type": "editorial_report",
            "status": status,
        })
