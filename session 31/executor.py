"""
Session 31 — executor.py

The Executor takes a validated Plan and runs the Steps in dependency order.

Compared to the S29 ReAct loop, the executor is deliberately DUMB:
  - It does not call the LLM between tool calls.
  - It does not decide what tool to use — the Plan already said.
  - It does not improvise — if a step fails it asks the monitor and
    follows the monitor's verdict.

This separation is the whole point of Plan-and-Execute:
  - The PLANNER is the smart, expensive call (a long-context LLM).
  - The EXECUTOR is cheap and predictable (a Python loop with a tool
    dispatcher).
  - The MONITOR is a small, focused LLM call that decides one bit:
    should the plan be revised?

Production patterns active here:
  - MAX-STEP LIMIT (PROD PATTERN, S31) — hard ceiling on steps actually
    executed, regardless of plan size or replan count.
  - PLACEHOLDER SUBSTITUTION — step args can reference {sN.observation}
    to pull data from earlier steps. The executor substitutes BEFORE
    calling the tool; the tool never sees the placeholder syntax.
  - DETERMINISTIC FAILURE RECORDING — every step produces a StepResult,
    whether it succeeded or not. Nothing silently disappears.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional

from plan_types import ExecutionResult, Plan, Step, StepResult
from tools import REGISTRY


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------


DEFAULT_MAX_STEPS = 25  # PROD PATTERN: hard ceiling on total step executions,
                       # including ones added by replans. 25 is enough room
                       # for two replans of a 7-step plan.


_PLACEHOLDER_RE = re.compile(r"\{(s\d+)\.observation\}")


# ----------------------------------------------------------------------------
# Executor
# ----------------------------------------------------------------------------


class Executor:
    """
    Runs a Plan and produces an ExecutionResult.

    The executor is a plain Python object — no LLM call inside its loop
    except the synthesis at the very end (and even that is delegated to
    a callback the caller can replace).
    """

    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_step_done: Optional[Callable[[Step, StepResult], None]] = None,
    ) -> None:
        self.max_steps = max_steps
        self.on_step_done = on_step_done or (lambda step, result: None)

    def run(self, plan: Plan) -> ExecutionResult:
        """
        Execute the plan in dependency order. Stop at max_steps.

        This entry point does NOT call the monitor or the replanner — that
        flow lives in the Orchestrator (see demo.py). Keep this loop
        single-purpose so it is easy to read.
        """

        result = ExecutionResult(plan=plan)
        observations: Dict[str, str] = {}
        order = _topological_order(plan)

        for step in order:
            if len(result.results) >= self.max_steps:
                result.results.append(StepResult(
                    step_id=step.id,
                    status="skipped",
                    observation=(
                        f"ERROR: max_steps={self.max_steps} reached before "
                        f"reaching step {step.id}."
                    ),
                ))
                self.on_step_done(step, result.results[-1])
                break

            # If any dependency failed, skip this step.
            failed_deps = [
                d for d in step.depends_on
                if observations.get(d, "").startswith("ERROR:")
            ]
            if failed_deps:
                result.results.append(StepResult(
                    step_id=step.id,
                    status="skipped",
                    observation=(
                        f"ERROR: skipped because dependency "
                        f"{','.join(failed_deps)} failed."
                    ),
                ))
                self.on_step_done(step, result.results[-1])
                continue

            # Substitute any {sN.observation} placeholders in args.
            resolved_args = _substitute_args(step.args, observations)

            # Look up the tool and run it.
            tool = REGISTRY.get(step.tool)
            if tool is None:
                step_result = StepResult(
                    step_id=step.id,
                    status="error",
                    observation=f"ERROR: unknown tool {step.tool!r}.",
                )
            else:
                step_result = _run_one_tool(tool.name, tool.run, resolved_args)

            observations[step.id] = step_result.observation
            result.results.append(step_result)
            self.on_step_done(step, step_result)

        return result


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _topological_order(plan: Plan) -> List[Step]:
    """
    Return steps in an order that respects depends_on.

    The validator already ensured the plan is a DAG, so we know this
    terminates. We use Kahn's algorithm — simpler to read than DFS-based
    topo sort and produces a stable order (steps with the same readiness
    keep their plan order).
    """

    indegree: Dict[str, int] = {s.id: len(s.depends_on) for s in plan.steps}
    by_id: Dict[str, Step] = {s.id: s for s in plan.steps}
    ready: List[Step] = [s for s in plan.steps if indegree[s.id] == 0]
    out: List[Step] = []

    while ready:
        step = ready.pop(0)  # FIFO so output mirrors plan order when possible
        out.append(step)
        for candidate in plan.steps:
            if step.id in candidate.depends_on:
                indegree[candidate.id] -= 1
                if indegree[candidate.id] == 0:
                    ready.append(by_id[candidate.id])

    if len(out) != len(plan.steps):
        # Should have been caught by validate_plan, but defensive.
        raise RuntimeError("cycle reached _topological_order — bug")
    return out


def _substitute_args(
    args: Dict[str, Any], observations: Dict[str, str]
) -> Dict[str, Any]:
    """
    Replace any {sN.observation} placeholder with the actual observation
    text from step sN. Operates on string values only — non-string values
    pass through untouched.
    """

    out: Dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str):
            out[key] = _PLACEHOLDER_RE.sub(
                lambda m: observations.get(m.group(1), m.group(0)), value
            )
        else:
            out[key] = value
    return out


def _run_one_tool(
    tool_name: str,
    tool_fn: Callable[[Dict[str, Any]], str],
    args: Dict[str, Any],
) -> StepResult:
    """Run one tool with timing and error capture."""

    start = time.time()
    try:
        observation = tool_fn(args)
    except Exception as exc:
        observation = f"ERROR: tool {tool_name} raised {type(exc).__name__}: {exc}"
    duration_ms = int((time.time() - start) * 1000)

    status = "error" if observation.startswith("ERROR:") else "ok"
    return StepResult(
        step_id="",  # filled in by caller
        status=status,
        observation=observation,
        duration_ms=duration_ms,
    )


# ----------------------------------------------------------------------------
# CLI smoke test — run an executor against a tiny hand-built plan.
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    from plan_types import Plan, Step

    hand_plan = Plan(
        goal="What is 12 times 7?",
        synthesis_hint="State the product as a sentence.",
        steps=[
            Step(
                id="s1",
                description="Multiply 12 and 7",
                tool="calculator",
                args={"expression": "12 * 7"},
            ),
        ],
    )

    def _printer(step: Step, result: StepResult) -> None:
        marker = "✓" if result.status == "ok" else "✗"
        print(f"  {marker} {step.id} {step.description}")
        print(f"     observation: {result.observation}")

    execution = Executor(on_step_done=_printer).run(hand_plan)
    print(f"\nDone — {execution.succeeded_steps}/{len(execution.results)} steps ok.")
