"""
Session 31 — replan.py

Produces a REVISED plan given the partial execution so far.

The replanner is essentially the planner with one extra input: the
results we already have. Two important properties:

  1. It SEES what has worked. If steps s1 and s2 succeeded, the new plan
     should not redo them — it should incorporate their observations and
     build on top.
  2. It KNOWS the latest reason for replanning (passed in by the
     orchestrator from the monitor's verdict). That reason becomes part
     of the prompt so the new plan can avoid the same failure mode.

The output goes through the same validator as the original plan. Same
contract, same guarantees. The executor cannot tell the difference between
an original plan and a replanned one — that's intentional.

Production patterns active here:
  - REPLAN BUDGET (see monitor.py) — the orchestrator enforces the cap,
    not the replanner. Replanning N+1 times is the orchestrator's choice.
  - CARRY OBSERVATIONS FORWARD — the new plan starts from what is known,
    not from scratch. This avoids paying tokens twice for work already
    done.
"""

from __future__ import annotations

from typing import Optional

from llm_client import LLMClient
from plan_types import ExecutionResult, Plan, PlanValidationError, validate_plan
from planner import _dict_to_plan, _extract_json
from tools import render_catalog, tool_names


REPLAN_SYSTEM_PROMPT = """You are a careful planning assistant REVISING a plan.

You are given:
  - the original USER GOAL,
  - the ORIGINAL PLAN you (or a previous planner) wrote,
  - the RESULTS so far (what succeeded, what failed),
  - the REASON FOR REPLAN — why the monitor flagged the current plan.

Your job is to PRODUCE A REVISED PLAN AS JSON. The revised plan should:
  1. Reuse observations from steps that already succeeded. Do not redo
     work that worked. If step s2 produced a useful value, reference it
     in the new plan's args or skip it entirely.
  2. Avoid repeating the failure named in REASON FOR REPLAN.
  3. Use only tools from the CATALOG.
  4. Keep the same JSON shape as the original plan:
     {
       "synthesis_hint": "...",
       "steps": [
         {"id": "...", "description": "...", "tool": "...",
          "args": {...}, "depends_on": [...]}
       ]
     }

Output the JSON only. No prose.
"""


def replan(
    execution: ExecutionResult,
    reason: str,
    client: Optional[LLMClient] = None,
    max_tokens: int = 1024,
) -> Plan:
    """
    Produce a revised Plan given the partial execution.

    Raises PlanValidationError if the revised plan is malformed.
    """

    client = client or LLMClient()

    original_plan_lines = "\n".join(
        f"  {s.id}: {s.description}  [tool={s.tool}, args={s.args}]"
        for s in execution.plan.steps
    )
    results_lines = "\n".join(
        f"  {r.step_id}: {r.status} -> {r.observation[:160]}"
        for r in execution.results
    )

    user = (
        f"CATALOG OF TOOLS:\n{render_catalog()}\n\n"
        f"USER GOAL: {execution.plan.goal}\n\n"
        f"ORIGINAL PLAN:\n{original_plan_lines}\n\n"
        f"RESULTS SO FAR:\n{results_lines}\n\n"
        f"REASON FOR REPLAN: {reason}\n\n"
        "Revise the plan and produce JSON now."
    )

    raw = client.complete(
        system=REPLAN_SYSTEM_PROMPT, user=user, max_tokens=max_tokens
    )

    plan_dict = _extract_json(raw)
    new_plan = _dict_to_plan(plan_dict, execution.plan.goal)
    validate_plan(new_plan, allowed_tools=tool_names())
    return new_plan


# ----------------------------------------------------------------------------
# CLI smoke test
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    from plan_types import ExecutionResult, Plan, Step, StepResult

    failed = ExecutionResult(
        plan=Plan(
            goal="Tell me about Alan Turing.",
            synthesis_hint="Two-sentence summary.",
            steps=[
                Step(
                    id="s1",
                    description="Look up Turing biography on the web",
                    tool="web_search",
                    args={"query": "alan turing biography"},
                ),
            ],
        ),
        results=[
            StepResult(
                step_id="s1",
                status="error",
                observation=(
                    "No high-confidence result for 'alan turing biography'."
                ),
            ),
        ],
        replan_count=0,
    )
    new_plan = replan(failed, reason="web_search returned no result; try wikipedia_summary")
    print(f"REVISED PLAN — {len(new_plan.steps)} steps")
    for s in new_plan.steps:
        print(f"  {s.id}: {s.description}  [tool={s.tool}]")
