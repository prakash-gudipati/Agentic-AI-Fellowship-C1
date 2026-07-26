"""
Session 35 — agents/director.py

PROD PATTERN: Hierarchical Decomposition (HEADLINE) — level 2 of 2.

The Director is the top-level orchestrator. It NEVER talks to workers
directly. It only talks to managers (research_manager,
editorial_manager). This is the discipline that makes the hierarchy
worth its overhead — every level of management is responsible for one
narrow kind of decision, and authority flows down, results flow up.

This is the same idea as S31 Plan-as-Artifact (the plan is auditable)
and S34 Orchestrator-Worker Decomposition (decomposition is a SKILL,
not a side effect) — applied one level higher.

PROD PATTERN: Termination Conditions still apply at the Director
layer. The Director has a max_director_rounds ceiling (default 4) to
guarantee the system halts.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

from agent_types import Fact, HierarchicalResult
from llm_client import LLMClient
from messages import MessageBus
from prompts import DIRECTOR_SYSTEM
from agents.research_manager import ResearchManager
from agents.editorial_manager import EditorialManager


MAX_DIRECTOR_ROUNDS = 4


class Director:
    def __init__(self, llm: LLMClient,
                 trace: Callable[[Dict], None],
                 bus: Optional[MessageBus] = None) -> None:
        self.llm = llm
        self.trace = trace
        self.bus = bus or MessageBus()
        self.research_manager = ResearchManager(llm, self.bus, trace)
        self.editorial_manager = EditorialManager(llm, self.bus, trace)

    # ------------------------------------------------------------------
    # Public entry point.
    # ------------------------------------------------------------------

    def run(self, user_request: str) -> HierarchicalResult:
        self.llm.reset_count()
        merged_facts: List[Fact] = []
        final_draft = ""
        terminated_reason = "max_rounds"

        for round_num in range(MAX_DIRECTOR_ROUNDS):
            decision = self._director_decide(user_request, merged_facts,
                                              final_draft)

            # Director can publish a final answer right away.
            if decision.get("done"):
                final_draft = decision.get("final", "") or final_draft
                terminated_reason = "all_done"
                self.trace({
                    "type": "termination",
                    "reason": terminated_reason,
                    "rounds": round_num + 1,
                    "calls": self.llm.call_count,
                })
                break

            next_mgr = decision.get("next_manager")
            instruction = decision.get("instruction", "")
            self.trace({
                "type": "director_decision",
                "done": False,
                "next_manager": next_mgr,
                "instruction": instruction,
            })

            # Send the ASSIGN message via the bus for audit.
            self.bus.assign(
                sender="director", recipient=next_mgr or "?",
                subject=instruction, round_num=round_num,
            )
            self.trace({
                "type": "message_sent",
                "sender": "director", "recipient": next_mgr or "?",
                "intent": "ASSIGN", "subject": instruction,
            })

            if next_mgr == "research_manager":
                merged_facts = self.research_manager.run(
                    user_request, round_num=round_num,
                )
                continue

            if next_mgr == "editorial_manager":
                draft, status = self.editorial_manager.run(
                    merged_facts, round_num=round_num,
                )
                final_draft = draft
                if status == "ESCALATED":
                    # Director will publish the escalated draft on the
                    # next loop iteration via the escalation branch.
                    terminated_reason = "escalated"
                continue

            # Unknown manager — terminate to avoid silent hangs.
            terminated_reason = "unknown_manager"
            break

        else:
            # Loop fell out — max rounds.
            self.trace({
                "type": "termination",
                "reason": terminated_reason,
                "rounds": MAX_DIRECTOR_ROUNDS,
                "calls": self.llm.call_count,
            })

        return HierarchicalResult(
            final_answer=final_draft,
            messages=self.bus.all_messages(),
            total_llm_calls=self.llm.call_count,
            terminated_reason=terminated_reason,
        )

    # ------------------------------------------------------------------
    # Director's per-round decision.
    # ------------------------------------------------------------------

    def _director_decide(self, user_request: str,
                          merged_facts: List[Fact],
                          final_draft: str) -> Dict:
        # Compose a structured prompt that includes the bus state hints
        # the FAKE_LLM router needs to dispatch correctly. In production
        # this would be the full message bus serialised.

        merged_count = len(merged_facts)
        merged_block = "(none yet)"
        if merged_facts:
            merged_block = json.dumps([
                {"text": f.text, "source": f.source}
                for f in merged_facts
            ])

        # Find latest editorial verdict if any.
        editorial_status = "pending"
        editorial_messages = self.bus.by_sender("editorial_manager")
        for m in reversed(editorial_messages):
            if m.recipient == "director" and m.intent == "REPORT":
                payload = m.payload or {}
                editorial_status = payload.get("status", "pending")
                break

        user = (
            f"USER_REQUEST: {user_request}\n"
            f"merged_facts_count={merged_count}\n"
            f"merged_facts: {merged_block}\n"
            f"editorial_report={editorial_status.lower()}\n"
            f"DRAFT:\n{final_draft}\n"
        )
        raw = self.llm.complete(DIRECTOR_SYSTEM, user)
        return _parse_director(raw)


def _parse_director(raw: str) -> Dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"done": True, "final": raw}
    if not isinstance(parsed, dict):
        return {"done": True, "final": str(parsed)}
    return parsed
