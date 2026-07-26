"""
Session 31 — plan_types.py

Data structures the planner, executor, monitor, and replan modules all share.

A Plan is not free-text. It is a structured object the executor can read,
the monitor can inspect, and the replan module can revise. That is the
single most important shift in this session: in S29's ReAct, the "plan"
existed only inside the LLM's head, one turn at a time. Here the plan is
a Python object you can print, validate, and walk through line by line.

Why this matters in production:
  - You can store the plan in a database alongside the goal, replay it,
    or diff two plans for the same goal.
  - A QA team can read the plan before execution and approve it.
  - The executor can be a different model (or even a script) — it does
    not need the reasoning power that produced the plan.

Production pattern introduced here (S31 new):
  - PLAN AS ARTIFACT — the plan is a typed, validated object, not a string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Step:
    """
    One unit of work in a Plan.

    Fields:
      id           : short identifier the executor uses to address this step
                     and that downstream steps reference in their args.
                     Convention: "s1", "s2", "s3", ...
      description  : human-readable line. The script prints this so an
                     instructor can read the plan to a student without
                     having to decode tool names.
      tool         : name of the tool to call. Must exist in the tool
                     registry the executor knows about — the planner is
                     told the tool catalog and must respect it.
      args         : dict passed to the tool. Values may contain references
                     like "{s1.observation}" — the executor substitutes
                     these at run time with the observation from step s1.
      depends_on   : step ids that must complete successfully before this
                     step can run. The executor walks the plan in
                     topological order based on this list.
    """

    id: str
    description: str
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class Plan:
    """
    A Plan is a goal plus an ordered list of Steps.

    The planner produces this. The executor consumes it.
    The monitor inspects it. The replan module can replace it.

    Fields:
      goal           : the original user question, verbatim. Kept so
                       downstream modules don't have to re-fetch it.
      steps          : the ordered list of Step objects.
      synthesis_hint : a short instruction the planner adds for the
                       final-answer synthesis step. Free-text. e.g.,
                       "compare the two per-capita figures and state which
                       is larger, with the percentage difference."
    """

    goal: str
    steps: List[Step] = field(default_factory=list)
    synthesis_hint: str = ""

    def by_id(self, step_id: str) -> Optional[Step]:
        """Look up a Step by its id, or None if missing."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None


@dataclass
class StepResult:
    """
    What happened when the executor ran one Step.

    status values:
      "ok"      — tool returned a normal string observation.
      "error"   — tool raised, or the executor refused to run (e.g., a
                  dependency failed). The observation explains why.
      "skipped" — step was skipped because a dependency failed and the
                  monitor decided continuing was not safe.
    """

    step_id: str
    status: str
    observation: str
    duration_ms: int = 0


@dataclass
class ExecutionResult:
    """
    The full record of one Plan execution.

    plan            : the Plan that was run.
    results         : one StepResult per step, in execution order.
    final_answer    : the synthesised answer to the original goal, or None
                      if the executor never reached synthesis.
    drift_detected  : True if the monitor flagged drift at least once.
    replan_count    : how many times the plan was revised mid-execution.
    total_tool_calls: count of "ok" + "error" StepResults (skips don't count).
    """

    plan: Plan
    results: List[StepResult] = field(default_factory=list)
    final_answer: Optional[str] = None
    drift_detected: bool = False
    replan_count: int = 0

    @property
    def total_tool_calls(self) -> int:
        return sum(1 for r in self.results if r.status in ("ok", "error"))

    @property
    def succeeded_steps(self) -> int:
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def failed_steps(self) -> int:
        return sum(1 for r in self.results if r.status == "error")


# ----------------------------------------------------------------------------
# Validation helpers — the planner module calls these before returning a Plan.
# ----------------------------------------------------------------------------


class PlanValidationError(Exception):
    """Raised when a plan does not satisfy the executor's preconditions."""


def validate_plan(plan: Plan, allowed_tools: List[str]) -> None:
    """
    Raise PlanValidationError if the plan is malformed.

    Checks:
      1. Every step references a tool the executor actually knows.
      2. Every depends_on entry points to a step id that exists.
      3. There are no cycles (the dependency graph is a DAG).
      4. Step ids are unique.

    The executor REFUSES to run a plan that fails these checks. This is the
    point of having a typed Plan in the first place — the planner is allowed
    to hallucinate, the validator is not.
    """

    if not plan.steps:
        raise PlanValidationError("plan has zero steps")

    ids = [s.id for s in plan.steps]
    if len(set(ids)) != len(ids):
        raise PlanValidationError(f"duplicate step ids in plan: {ids}")

    for step in plan.steps:
        if step.tool not in allowed_tools:
            raise PlanValidationError(
                f"step {step.id} references unknown tool {step.tool!r}; "
                f"allowed: {allowed_tools}"
            )
        for dep in step.depends_on:
            if dep not in ids:
                raise PlanValidationError(
                    f"step {step.id} depends on {dep!r}, which does not exist"
                )

    _check_no_cycles(plan)


def _check_no_cycles(plan: Plan) -> None:
    """Standard DFS cycle check on the depends_on graph."""

    WHITE, GRAY, BLACK = 0, 1, 2
    colour = {s.id: WHITE for s in plan.steps}
    graph = {s.id: list(s.depends_on) for s in plan.steps}

    def visit(node: str, stack: List[str]) -> None:
        if colour[node] == GRAY:
            raise PlanValidationError(
                f"cycle in plan dependencies: {' -> '.join(stack + [node])}"
            )
        if colour[node] == BLACK:
            return
        colour[node] = GRAY
        for nxt in graph[node]:
            visit(nxt, stack + [node])
        colour[node] = BLACK

    for s in plan.steps:
        if colour[s.id] == WHITE:
            visit(s.id, [])
