"""
Session 31 — monitor.py

The Monitor decides ONE BIT after each step: should we replan?

Why a dedicated module:
  - The executor is intentionally dumb. It runs steps. It does not judge.
  - The planner is expensive. We do not want to invoke a replan unless
    something is actually wrong.
  - The monitor is a small focused LLM call that reads the latest step
    result + the remaining plan and returns:
        {"should_replan": bool, "reason": str}

Heuristic short-circuits run BEFORE the LLM call:
  - If the latest step succeeded ("ok" status), no replan needed.
  - If we're already past the replan budget, no replan even if we'd want one.
  - If the failure rate exceeds 60% of executed steps, replan
    unconditionally — the plan is broken at a structural level.

If none of those fire, we ask the LLM. The fake-mode path returns
"should_replan: true" whenever an error or no-result observation is
present, which mirrors what Claude does on this prompt in practice.

Production patterns active here:
  - REPLAN BUDGET (PROD PATTERN, S31) — agents that replan forever are
    expensive and rarely converge. Cap it at 2 replans per goal.
  - HEURISTIC-BEFORE-LLM — every cheap check we can do in Python comes
    BEFORE the LLM call. Saves tokens and latency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from llm_client import LLMClient
from plan_types import ExecutionResult, Plan


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------


DEFAULT_REPLAN_BUDGET = 2  # PROD PATTERN: hard cap on replans per goal.
DEFAULT_FAILURE_RATE_REPLAN = 0.6  # 60% of executed steps failing -> replan.


# ----------------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------------


@dataclass
class MonitorDecision:
    """
    The verdict returned by Monitor.evaluate().

    should_replan : if True, the orchestrator hands the partial plan +
                    results to the replanner; if False, execution continues.
    reason        : human-readable explanation. Always populated.
    source        : "heuristic" | "llm" | "budget" — useful for traces so
                    a student can see which path the decision took.
    """

    should_replan: bool
    reason: str
    source: str = "heuristic"


# ----------------------------------------------------------------------------
# Monitor
# ----------------------------------------------------------------------------


MONITOR_SYSTEM_PROMPT = """You are the MONITOR for a Plan-and-Execute agent.

You read:
  - the original USER GOAL,
  - the PLAN the agent is executing,
  - the LATEST STEP RESULT (the most recent observation),
  - a summary of HOW MANY STEPS have succeeded vs failed so far.

You return a SINGLE JSON object with two fields:
  {"should_replan": <true|false>, "reason": "<one short sentence>"}

Decide should_replan = TRUE only when at least one of these is true:
  1. The latest observation starts with "ERROR:" and is essential to the
     remaining plan (i.e., later steps depend on it).
  2. The latest observation is a "no result found" message AND the goal
     cannot be answered without that piece of information.
  3. The plan is clearly mis-targeted at this goal (the wrong tool was
     chosen and is producing junk).

If the latest step succeeded, return should_replan = false.
If the failure is recoverable (later steps don't depend on it), return false.

Output the JSON object only. No prose.
"""


class Monitor:
    """
    Inspects an ExecutionResult after each step and decides whether to
    trigger a replan.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        replan_budget: int = DEFAULT_REPLAN_BUDGET,
        failure_rate_threshold: float = DEFAULT_FAILURE_RATE_REPLAN,
    ) -> None:
        self.client = client or LLMClient()
        self.replan_budget = replan_budget
        self.failure_rate_threshold = failure_rate_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, execution: ExecutionResult) -> MonitorDecision:
        """Look at the latest step and return a decision."""

        if not execution.results:
            return MonitorDecision(False, "no steps executed yet", "heuristic")

        # PROD PATTERN: enforce the replan budget BEFORE asking anything else.
        if execution.replan_count >= self.replan_budget:
            return MonitorDecision(
                False,
                f"replan budget exhausted ({self.replan_budget})",
                "budget",
            )

        latest = execution.results[-1]

        # Heuristic 1 — last step succeeded => no replan needed.
        if latest.status == "ok":
            return MonitorDecision(False, "latest step succeeded", "heuristic")

        # Heuristic 2 — high failure rate => structural problem, replan now.
        total_executed = max(1, execution.total_tool_calls)
        failure_rate = execution.failed_steps / total_executed
        if failure_rate >= self.failure_rate_threshold:
            return MonitorDecision(
                True,
                f"failure rate {failure_rate:.0%} above threshold",
                "heuristic",
            )

        # Otherwise, ask the LLM for a focused verdict.
        return self._ask_llm(execution)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ask_llm(self, execution: ExecutionResult) -> MonitorDecision:
        latest = execution.results[-1]
        plan_lines = "\n".join(
            f"  {s.id}: {s.description}  [tool={s.tool}]"
            for s in execution.plan.steps
        )
        results_lines = "\n".join(
            f"  {r.step_id}: {r.status} -> {r.observation[:120]}"
            for r in execution.results
        )

        user = (
            "You are the MONITOR — judge whether the agent should revise its plan.\n\n"
            f"USER GOAL: {execution.plan.goal}\n\n"
            f"PLAN:\n{plan_lines}\n\n"
            f"RESULTS SO FAR:\n{results_lines}\n\n"
            f"LATEST OBSERVATION: {latest.observation}\n\n"
            "Return your decision as JSON now."
        )

        raw = self.client.complete(
            system=MONITOR_SYSTEM_PROMPT, user=user, max_tokens=200
        )

        try:
            data = json.loads(_extract_first_json(raw))
        except Exception:
            # If the monitor itself misbehaves, default to "do not replan"
            # — it's safer to let the orchestrator continue and possibly
            # surface a clean ERROR than to spin in replans.
            return MonitorDecision(
                False,
                f"monitor LLM unparseable; defaulting to continue: {raw[:80]!r}",
                "llm",
            )

        return MonitorDecision(
            should_replan=bool(data.get("should_replan", False)),
            reason=str(data.get("reason", "")),
            source="llm",
        )


def _extract_first_json(raw: str) -> str:
    """Pull the first {...} block out of a possibly-chatty LLM response."""

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return "{}"
    return raw[start : end + 1]


# ----------------------------------------------------------------------------
# CLI smoke test
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    from plan_types import ExecutionResult, Plan, Step, StepResult

    # Simulate a failed first step on the Turing lookup goal.
    plan = Plan(
        goal="Tell me about Alan Turing.",
        synthesis_hint="Two-sentence summary.",
        steps=[
            Step(id="s1", description="Look up Turing biography",
                 tool="web_search", args={"query": "alan turing biography"}),
        ],
    )
    execution = ExecutionResult(
        plan=plan,
        results=[
            StepResult(
                step_id="s1",
                status="error",
                observation=(
                    "No high-confidence result for 'alan turing biography'. "
                    "Try a more specific query or use the wikipedia_summary "
                    "tool if you are looking for a definition."
                ),
            )
        ],
        replan_count=0,
    )

    decision = Monitor().evaluate(execution)
    print(f"should_replan={decision.should_replan}")
    print(f"reason={decision.reason}")
    print(f"source={decision.source}")
