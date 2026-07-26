"""
Session 34 — agents/orchestrator.py

The orchestrator. One job: decompose the user's request, decide which
worker runs next, stop when the work is done. PROD PATTERN:
Orchestrator-Worker Decomposition — the headline pattern of S34.

Termination conditions (PROD PATTERN: Termination Conditions):
  - QUALITY-MET: critique verdict ACCEPT → emit FINAL_ANSWER, stop.
  - ALL-DONE: orchestrator returns done=true → stop.
  - MAX-ROUNDS: hard ceiling (default 5) — stop even if not done.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent_types import CrewResult
from agents.fact_checker import FactChecker
from agents.researcher import Researcher
from agents.writer import Writer
from llm_client import LLMClient
from prompts import ORCHESTRATOR_SYSTEM
from scratchpad import (
    SECTION_CRITIQUE,
    SECTION_DRAFT,
    SECTION_FACTS,
    SECTION_FINAL,
    Scratchpad,
)


DEFAULT_MAX_ROUNDS = 5
KNOWN_WORKERS = ("researcher", "writer", "fact_checker")


@dataclass
class Orchestrator:
    llm: Optional[LLMClient] = None
    max_rounds: int = DEFAULT_MAX_ROUNDS
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None

    # Worker instances — caller can swap these for testing.
    researcher: Optional[Researcher] = None
    writer: Optional[Writer] = None
    fact_checker: Optional[FactChecker] = None

    # For Demo 4 — inject a marker the writer plants on round 1 so the
    # fact-checker has something to flag.
    writer_inject_marker: str = ""

    def __post_init__(self) -> None:
        self.llm = self.llm or LLMClient()
        self.researcher = self.researcher or Researcher(llm=self.llm)
        self.writer = self.writer or Writer(llm=self.llm)
        self.fact_checker = self.fact_checker or FactChecker(llm=self.llm)
        self.name = "orchestrator"

    # ----------------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------------

    def run(self, user_request: str) -> CrewResult:
        started = time.perf_counter()
        scratchpad = Scratchpad()
        rounds = 0
        terminated_reason = ""

        for round_no in range(1, self.max_rounds + 1):
            rounds = round_no
            self._emit("round_start", {"round_no": round_no})

            decision = self._decide_next(user_request, scratchpad)
            self._emit("orchestrator_decision", decision)

            if decision.get("done"):
                final = decision.get("final", "") or scratchpad.get_draft()
                scratchpad.write(
                    SECTION_FINAL,
                    final,
                    agent=self.name,
                    round_no=round_no,
                    summary="final answer published",
                )
                terminated_reason = "all_done"
                break

            worker = decision.get("worker")
            instruction = decision.get("instruction", "")

            if worker not in KNOWN_WORKERS:
                # Orchestrator hallucination — surface as text, end.
                scratchpad.append_to_notes(
                    f"unknown worker '{worker}' — terminating",
                    agent=self.name,
                    round_no=round_no,
                )
                terminated_reason = "unknown_worker"
                break

            self._dispatch_worker(worker, instruction, scratchpad, round_no)

            # Quality-met termination — if the critique just landed and
            # accepts, publish FINAL_ANSWER and stop.
            critique = scratchpad.get_critique()
            if critique is not None and critique.accepted:
                final = scratchpad.get_draft()
                scratchpad.write(
                    SECTION_FINAL,
                    final,
                    agent=self.name,
                    round_no=round_no,
                    summary="final answer (quality_met)",
                )
                terminated_reason = "quality_met"
                self._emit(
                    "critique_verdict",
                    {"accepted": True},
                )
                break
            if critique is not None and not critique.accepted:
                self._emit(
                    "critique_verdict",
                    {"accepted": False, "issues": critique.issues},
                )

        if not terminated_reason:
            terminated_reason = "max_rounds"
            # Publish best draft we have so demos always emit something.
            scratchpad.write(
                SECTION_FINAL,
                scratchpad.get_draft() or "(no draft produced)",
                agent=self.name,
                round_no=rounds,
                summary="final published under max_rounds ceiling",
            )

        self._emit(
            "termination",
            {"reason": terminated_reason, "rounds": rounds},
        )

        elapsed = time.perf_counter() - started
        return CrewResult(
            user_request=user_request,
            final_text=scratchpad.get_final(),
            rounds=rounds,
            max_rounds=self.max_rounds,
            terminated_reason=terminated_reason,
            facts=scratchpad.get_facts(),
            critique=scratchpad.get_critique(),
            elapsed_seconds=elapsed,
            llm_calls=self.llm.call_count,
        )

    # ----------------------------------------------------------------
    # Decision step — single LLM call
    # ----------------------------------------------------------------

    def _decide_next(
        self, user_request: str, scratchpad: Scratchpad
    ) -> Dict[str, Any]:
        """Call the orchestrator LLM with a snapshot of the scratchpad
        and parse the JSON decision."""

        facts = scratchpad.get_facts()
        draft = scratchpad.get_draft()
        critique = scratchpad.get_critique()

        facts_block = (
            "[]" if not facts
            else "[" + ", ".join(
                f"{{text=\"{f.text}\", source=\"{f.source}\"}}" for f in facts
            ) + "]"
        )
        critique_block = (
            "(empty)" if critique is None
            else f"verdict={critique.verdict} issues={json.dumps(critique.issues)} notes=\"{critique.notes}\""
        )

        user_prompt = (
            f"USER_REQUEST: {user_request}\n\n"
            f"SCRATCHPAD STATE:\n"
            f"  FACTS: {facts_block}\n"
            f"  DRAFT: {draft or '(empty)'}\n"
            f"  CRITIQUE: {critique_block}\n\n"
            "Emit the next decision as JSON: "
            "{\"next_worker\": \"<name>\", \"instruction\": \"<sentence>\"} "
            "or {\"done\": true, \"final\": \"<text>\"}."
        )

        raw = self.llm.complete(
            ORCHESTRATOR_SYSTEM, user_prompt, max_tokens=400
        )
        decision = _parse_decision(raw)
        # Normalise key names.
        return {
            "done": bool(decision.get("done", False)),
            "worker": decision.get("next_worker") or decision.get("worker"),
            "instruction": decision.get("instruction", ""),
            "final": decision.get("final", ""),
        }

    # ----------------------------------------------------------------
    # Worker dispatch
    # ----------------------------------------------------------------

    def _dispatch_worker(
        self,
        worker: str,
        instruction: str,
        scratchpad: Scratchpad,
        round_no: int,
    ) -> None:
        self._emit("worker_started", {"worker": worker})

        if worker == "researcher":
            facts = self.researcher.run(instruction, scratchpad, round_no)
            self._emit_scratchpad(
                "researcher", SECTION_FACTS,
                f"{len(facts)} facts gathered",
            )
            self._emit(
                "worker_finished",
                {"worker": worker, "summary": f"{len(facts)} facts"},
            )
        elif worker == "writer":
            # Demo marker logic:
            #  - ADVERSARIAL_QUESTION: inject every time (Demo 5 needs the
            #    fact-checker to keep rejecting until the budget runs out).
            #  - INJECT_BAD_CLAIM: inject only on the writer's FIRST call;
            #    on subsequent calls the writer "revises" without the marker
            #    so the fact-checker can accept (Demo 4).
            marker = ""
            if self.writer_inject_marker == "ADVERSARIAL_QUESTION":
                marker = "ADVERSARIAL_QUESTION"
            elif self.writer_inject_marker and not scratchpad.get_draft():
                marker = self.writer_inject_marker
            draft = self.writer.run(
                instruction, scratchpad, round_no, inject_marker=marker,
            )
            # The writer just produced a fresh draft — any prior critique
            # is stale. Clear it so the orchestrator routes to fact_checker
            # again on the next round rather than treating REVISE as still
            # active.
            scratchpad.sections.pop(SECTION_CRITIQUE, None)
            self._emit_scratchpad(
                "writer", SECTION_DRAFT, f"draft len={len(draft)} chars",
            )
            self._emit(
                "worker_finished",
                {"worker": worker, "summary": f"draft {len(draft)} chars"},
            )
        elif worker == "fact_checker":
            critique = self.fact_checker.run(instruction, scratchpad, round_no)
            self._emit_scratchpad(
                "fact_checker", SECTION_CRITIQUE,
                f"verdict={critique.verdict}",
            )
            self._emit(
                "worker_finished",
                {"worker": worker, "summary": f"verdict={critique.verdict}"},
            )

    # ----------------------------------------------------------------
    # Trace plumbing
    # ----------------------------------------------------------------

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(kind, payload)

    def _emit_scratchpad(self, agent: str, section: str, summary: str) -> None:
        self._emit(
            "scratchpad_write",
            {"agent": agent, "section": section, "summary": summary},
        )


def _parse_decision(raw: str) -> Dict[str, Any]:
    """Pull the first JSON object out of the orchestrator's output."""

    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0:
        return {}
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
