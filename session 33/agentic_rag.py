"""
Session 33 — agentic_rag.py

The agentic RAG loop. This file is the centre of the session.

What's different from S30's tool-calling agent?
  - The tool surface is tiny (search_kb, list_documents) — the
    interesting decisions are about RETRIEVAL DISCIPLINE.
  - Every search_kb result is automatically scored by the quality gate
    before the agent sees it. The gate's pass/fail flag is part of the
    tool_result the agent reads on the next turn.
  - A retrieval BUDGET caps how many times the agent can call search_kb
    per user turn. Once the budget hits zero, the agent must synthesise
    from whatever it has.

What's different from S29's ReAct loop?
  - Native function calling (via llm_client.decide_next), not regex
    parsing.
  - Synthesis is an explicit step at the end, not the same call that
    decides the next action.

Four PROD PATTERNS introduced or reused in this file:
  - Retrieval as a Tool      (reused from S30)
  - Retrieval Quality Gate   (NEW — the curriculum's headline for S33)
  - Multi-Hop Retrieval      (NEW — re-query on gate failure)
  - Retrieval Budget         (NEW — extends S31's max-step ceiling)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from llm_client import AgentDecision, LLMClient
from prompts import AGENTIC_RAG_SYSTEM
from quality_gate import GateVerdict, evaluate
from query_rewriter import decompose
from rag_types import AgenticAnswer, RetrievalAttempt, RetrievedChunk
from retrieval_tools import TOOL_SCHEMAS
from tools import ToolResult, dispatch


# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------


DEFAULT_RETRIEVAL_BUDGET = 3
DEFAULT_QUALITY_THRESHOLD = 3.5
MAX_LOOP_ITERATIONS = 8


# ----------------------------------------------------------------------------
# AgenticRAG
# ----------------------------------------------------------------------------


@dataclass
class AgenticRAG:
    """One stateless answerer. Construct once per process and reuse."""

    llm: Optional[LLMClient] = None
    retrieval_budget: int = DEFAULT_RETRIEVAL_BUDGET
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
    use_decomposition: bool = True
    use_quality_gate: bool = True
    # Hook the trace logger plugs into. demo.py supplies a real logger;
    # tests leave it as None.
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None

    def __post_init__(self) -> None:
        self.llm = self.llm or LLMClient()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def answer(self, user_question: str) -> AgenticAnswer:
        started = time.perf_counter()
        self._emit("user_question", {"text": user_question})

        # STEP A — Decomposition short-circuit.
        # If the question is compound, split it. Run the agent loop for
        # each sub-question and merge the evidence at the end. This is
        # the Decompose-then-Retrieve PROD PATTERN.
        sub_questions: List[str] = [user_question.strip()]
        if self.use_decomposition:
            sub_questions = decompose(user_question, llm=self.llm)
            if len(sub_questions) > 1:
                self._emit(
                    "decomposed", {"sub_questions": sub_questions}
                )

        all_attempts: List[RetrievalAttempt] = []
        budget_remaining = self.retrieval_budget
        any_budget_exhausted = False
        decided_no_retrieval = False

        for sq_idx, sq in enumerate(sub_questions):
            if len(sub_questions) > 1:
                self._emit(
                    "sub_question_start",
                    {"index": sq_idx + 1, "text": sq},
                )

            sub_attempts, budget_remaining, no_retrieval = self._run_loop(
                user_question=user_question,
                sub_question=sq,
                budget_remaining=budget_remaining,
            )
            all_attempts.extend(sub_attempts)
            if no_retrieval:
                decided_no_retrieval = True
            if budget_remaining <= 0 and any(
                not a.gate_passed for a in sub_attempts
            ):
                any_budget_exhausted = True

        # STEP B — Final synthesis pass.
        # All retrievals are done; ask the LLM to write the answer from
        # whatever the loop collected. If the agent already produced a
        # final answer in the loop (no-retrieval branch), reuse it.
        final_text = self._synthesise(
            user_question=user_question,
            attempts=all_attempts,
            decided_no_retrieval=decided_no_retrieval,
        )

        elapsed = time.perf_counter() - started
        return AgenticAnswer(
            user_question=user_question,
            answer_text=final_text,
            attempts=all_attempts,
            retrieval_budget=self.retrieval_budget,
            budget_exhausted=any_budget_exhausted,
            decided_no_retrieval=decided_no_retrieval,
            decomposed_sub_questions=sub_questions
            if len(sub_questions) > 1
            else [],
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # The agent loop, scoped to one sub-question
    # ------------------------------------------------------------------

    def _run_loop(
        self,
        user_question: str,
        sub_question: str,
        budget_remaining: int,
    ):
        """Drive decide_next until either:
        - the agent emits a final answer (no_retrieval = True)
        - the budget hits zero
        - the agent calls search_kb and we score the result

        Returns the per-sub-question attempts, the updated budget, and
        a flag for the no-retrieval path.
        """

        attempts: List[RetrievalAttempt] = []

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": sub_question}
        ]
        attempt_index = 0
        no_retrieval = False

        for _ in range(MAX_LOOP_ITERATIONS):
            decision = self.llm.decide_next(
                system=AGENTIC_RAG_SYSTEM,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
            self._emit(
                "decision",
                {
                    "kind": decision.kind,
                    "tool_name": decision.tool_name,
                    "tool_args": decision.tool_args,
                },
            )

            if decision.kind == "final":
                # The agent decided no retrieval was needed.
                if attempt_index == 0:
                    no_retrieval = True
                    self._emit(
                        "no_retrieval_decided", {"answer": decision.text}
                    )
                    # Stash the final text on a synthetic attempt so the
                    # synthesise pass can pick it up.
                    attempts.append(
                        RetrievalAttempt(
                            attempt_index=0,
                            query="(none — answered directly)",
                            chunks=[],
                            average_relevance=None,
                            gate_passed=True,
                            notes="no-retrieval path: " + decision.text,
                        )
                    )
                break

            # Tool call path.
            if decision.tool_name != "search_kb":
                # list_documents is cheap, doesn't burn budget.
                tool_result = dispatch(
                    decision.tool_name, decision.tool_args
                )
                messages.extend(
                    _append_assistant_and_tool_result(
                        decision, tool_result
                    )
                )
                continue

            if budget_remaining <= 0:
                # Hard cap. The agent must synthesise now.
                self._emit(
                    "budget_exhausted",
                    {"sub_question": sub_question},
                )
                break

            attempt_index += 1
            budget_remaining -= 1

            # Run the retrieval.
            query = str(decision.tool_args.get("query") or "")
            tool_result = dispatch("search_kb", decision.tool_args)
            chunks: List[RetrievedChunk] = (
                tool_result.payload if tool_result.ok else []
            )

            # Score chunks through the quality gate.
            verdict: Optional[GateVerdict] = None
            if self.use_quality_gate and chunks:
                verdict = evaluate(
                    query=query,
                    chunks=chunks,
                    llm=self.llm,
                    threshold=self.quality_threshold,
                )
                self._emit(
                    "gate_verdict",
                    {
                        "query": query,
                        "average": verdict.average_relevance,
                        "passed": verdict.passed,
                    },
                )

            attempt = RetrievalAttempt(
                attempt_index=attempt_index,
                query=query,
                chunks=chunks,
                average_relevance=verdict.average_relevance
                if verdict
                else None,
                gate_passed=verdict.passed if verdict else bool(chunks),
                notes=verdict.short() if verdict else "",
            )
            attempts.append(attempt)
            self._emit(
                "retrieval_attempt",
                {
                    "index": attempt.attempt_index,
                    "query": attempt.query,
                    "chunks": [c.short() for c in chunks],
                    "verdict": attempt.notes,
                    "budget_remaining": budget_remaining,
                },
            )

            # Feed the scored chunks back to the agent.
            messages.extend(
                _append_assistant_and_tool_result(
                    decision,
                    tool_result,
                    extra_note=attempt.notes,
                )
            )

            # If the gate passed, give the agent a chance to wrap up
            # next iteration. If the gate failed, the agent will see
            # the failure note in the tool_result and likely re-query.
            if not attempt.gate_passed and budget_remaining > 0:
                self._emit(
                    "gate_failed_triggers_requery",
                    {"budget_remaining": budget_remaining},
                )

        return attempts, budget_remaining, no_retrieval

    # ------------------------------------------------------------------
    # Final synthesis
    # ------------------------------------------------------------------

    def _synthesise(
        self,
        user_question: str,
        attempts: List[RetrievalAttempt],
        decided_no_retrieval: bool,
    ) -> str:
        # No-retrieval path: return the direct answer the agent already
        # produced. (Stashed in attempt[0].notes by _run_loop.)
        if decided_no_retrieval and attempts:
            note = attempts[0].notes
            if note.startswith("no-retrieval path: "):
                return note[len("no-retrieval path: ") :]

        # Collect every chunk that passed the gate (or all chunks if the
        # gate is disabled).
        evidence_blocks: List[str] = []
        for a in attempts:
            for c in a.chunks:
                if c.relevance is None or c.relevance >= 3:
                    evidence_blocks.append(f"[{c.source}] {c.text}")
        if not evidence_blocks:
            return (
                "I could not retrieve evidence strong enough to answer "
                "this question. (Retrieval budget exhausted or all "
                "chunks failed the quality gate.)"
            )

        evidence = "\n\n".join(evidence_blocks)
        synthesis_prompt = (
            f"USER_QUESTION:\n{user_question}\n\n"
            f"EVIDENCE:\n{evidence}\n\n"
            "Write a 2-4 sentence answer using only the evidence. Cite "
            "source filenames in parentheses. If the evidence does not "
            "fully answer the question, say what is missing."
        )
        return self.llm.complete(
            AGENTIC_RAG_SYSTEM, synthesis_prompt, max_tokens=350
        )

    # ------------------------------------------------------------------
    # Trace plumbing
    # ------------------------------------------------------------------

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(kind, payload)


# ----------------------------------------------------------------------------
# Message-list plumbing
# ----------------------------------------------------------------------------


def _append_assistant_and_tool_result(
    decision: AgentDecision,
    tool_result: ToolResult,
    extra_note: str = "",
) -> List[Dict[str, Any]]:
    """Build the two messages that close a tool-call round-trip.

    Anthropic's API expects:
      assistant: [text?, tool_use]
      user:      [tool_result]
    """

    assistant_blocks: List[Dict[str, Any]] = []
    if decision.text:
        assistant_blocks.append({"type": "text", "text": decision.text})
    assistant_blocks.append(
        {
            "type": "tool_use",
            "id": decision.tool_use_id or "call_unknown",
            "name": decision.tool_name,
            "input": decision.tool_args,
        }
    )

    payload_text = tool_result.to_text()
    if extra_note:
        payload_text = f"{payload_text}\n\n{extra_note}"

    user_blocks: List[Dict[str, Any]] = [
        {
            "type": "tool_result",
            "tool_use_id": decision.tool_use_id or "call_unknown",
            "content": payload_text,
            "is_error": not tool_result.ok,
        }
    ]
    return [
        {"role": "assistant", "content": assistant_blocks},
        {"role": "user", "content": user_blocks},
    ]
