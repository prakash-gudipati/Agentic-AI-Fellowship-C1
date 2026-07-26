"""
Session 31 — demo.py

Five demos that exercise the Plan-and-Execute system and contrast it with
the S29-style ReAct loop.

Usage:
    python demo.py 1     # ReAct baseline on the per-capita-GDP goal
    python demo.py 2     # Plan-and-Execute on the same goal
    python demo.py 3     # Side-by-side comparison (token + call counts)
    python demo.py 4     # Drift + replan: web_search misses, monitor
                         # triggers replan to wikipedia_summary
    python demo.py 5     # Max-step ceiling kicks in on a bad plan

By default these demos make REAL Anthropic API calls. The key is loaded
automatically from a .env file sitting next to this script (Session_31/Code/.env)
— just drop your ANTHROPIC_API_KEY in there. You can also export it in your
shell; either works. If you want the offline canned-response path (useful for
a flaky network or deterministic replay), prefix with FAKE_LLM=1, e.g.:

    FAKE_LLM=1 python demo.py 3
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ----------------------------------------------------------------------------
# Tiny .env loader (no python-dotenv dependency).
# Reads KEY=VALUE or KEY = VALUE lines from a .env file sitting next to this
# script and pushes them into os.environ if not already set. Lets students run
# `python demo.py 3` without manually exporting ANTHROPIC_API_KEY first.
# ----------------------------------------------------------------------------
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).parent / ".env")


from executor import Executor
from llm_client import LLMClient
from monitor import Monitor
from plan_types import ExecutionResult, Plan, PlanValidationError, Step, StepResult
from planner import make_plan
from react_baseline import run_react
from replan import replan
from tools import REGISTRY, render_catalog, tool_names
from trace_logger import (
    print_block,
    print_decision,
    print_plan,
    print_step_event,
)


# DEFAULT: REAL Anthropic API calls. Requires ANTHROPIC_API_KEY in the env.
# To re-enable offline canned-response mode (useful when the network is flaky
# or you want a deterministic replay), set FAKE_LLM=1 in your shell.
#
#   real run :   python demo.py 3
#   fake run :   FAKE_LLM=1 python demo.py 3
#
if os.environ.get("FAKE_LLM", "") != "1" and not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit(
        "\nERROR: ANTHROPIC_API_KEY is not set in your environment.\n"
        "       Either:\n"
        "         1) export ANTHROPIC_API_KEY=sk-ant-...   (real calls)\n"
        "         2) export FAKE_LLM=1                     (offline canned mode)\n"
        "       then re-run.\n"
    )


GOAL_PER_CAPITA = (
    "Compare India and United States per-capita GDP in 2024 and tell me "
    "which is larger plus the ratio."
)
GOAL_TURING = "Tell me about Alan Turing in two sentences."


# ----------------------------------------------------------------------------
# Demo 1 — ReAct baseline
# ----------------------------------------------------------------------------


def demo_1_react_baseline() -> None:
    print_block(
        "DEMO 1 — ReAct baseline on the per-capita-GDP goal",
        "S29-style loop: model decides what to do turn by turn.",
    )
    print(f"\nGOAL:\n  {GOAL_PER_CAPITA}\n")
    run = run_react(GOAL_PER_CAPITA)
    print(f"\nFINAL ANSWER:\n{run.text}")
    print(
        f"\nRUN STATS: turns={run.turns}  tool_calls={run.tool_call_count}"
    )


# ----------------------------------------------------------------------------
# Demo 2 — Plan-and-Execute, no drift
# ----------------------------------------------------------------------------


def demo_2_plan_and_execute() -> None:
    print_block(
        "DEMO 2 — Plan-and-Execute on the per-capita-GDP goal",
        "Planner writes a 7-step plan; executor runs each step in order; "
        "synthesiser composes the final answer.",
    )
    print(f"\nGOAL:\n  {GOAL_PER_CAPITA}\n")
    client = LLMClient()
    plan = make_plan(GOAL_PER_CAPITA, client=client)
    print_plan(plan)

    print()
    execution = Executor(on_step_done=print_step_event).run(plan)

    final = _synthesise(execution, client)
    execution.final_answer = final

    print(f"\nFINAL ANSWER:\n{final}")
    print(
        f"\nRUN STATS: "
        f"tool_calls={execution.total_tool_calls}  "
        f"replans={execution.replan_count}"
    )


# ----------------------------------------------------------------------------
# Demo 3 — side-by-side comparison
# ----------------------------------------------------------------------------


def demo_3_side_by_side() -> None:
    print_block(
        "DEMO 3 — ReAct vs Plan-and-Execute, same goal",
        "Compare turn count and tool-call count on the per-capita-GDP goal.",
    )
    print(f"\nGOAL:\n  {GOAL_PER_CAPITA}\n")

    print("\n--- ReAct ---")
    react_run = run_react(GOAL_PER_CAPITA)
    print(f"Final: {react_run.text[:120]}...")
    print(
        f"Stats: turns={react_run.turns}  "
        f"tool_calls={react_run.tool_call_count}"
    )

    print("\n--- Plan-and-Execute ---")
    client = LLMClient()
    plan = make_plan(GOAL_PER_CAPITA, client=client)
    execution = Executor().run(plan)
    final = _synthesise(execution, client)
    execution.final_answer = final
    print(f"Final: {final[:120]}...")
    print(
        f"Stats: planner_calls=1  tool_calls={execution.total_tool_calls}  "
        f"synthesis_calls=1  replans={execution.replan_count}"
    )

    print("\n--- The story this tells ---")
    print(
        "  ReAct ran "
        f"{react_run.turns} model round-trips for the same answer.\n"
        "  Plan-and-Execute ran 2 model round-trips (planner + synthesis)\n"
        "  plus the deterministic Python loop that ran the tools.\n"
        "  For a 7-step task this is roughly 4x fewer model calls."
    )


# ----------------------------------------------------------------------------
# Demo 4 — drift + replan
# ----------------------------------------------------------------------------


def demo_4_drift_and_replan() -> None:
    print_block(
        "DEMO 4 — drift and replan",
        "Planner picks the wrong tool (web_search for a biography). "
        "Monitor catches the failure. Replanner switches to wikipedia_summary.",
    )
    print(f"\nGOAL:\n  {GOAL_TURING}\n")

    client = LLMClient()
    plan = make_plan(GOAL_TURING, client=client)
    print_plan(plan)

    monitor = Monitor(client=client)
    execution = ExecutionResult(plan=plan)
    executor = Executor()

    for attempt in range(monitor.replan_budget + 1):
        print(f"\n--- attempt {attempt + 1} ---")
        partial = executor.run(execution.plan)
        # Carry results forward into the long-running ExecutionResult.
        execution.results.extend(partial.results)
        for step, result in zip(partial.plan.steps, partial.results):
            print_step_event(step, result)

        decision = monitor.evaluate(execution)
        print_decision(decision)
        if not decision.should_replan:
            break

        try:
            new_plan = replan(
                execution, reason=decision.reason, client=client
            )
        except PlanValidationError as exc:
            print(f"  replan failed validation: {exc}")
            break

        execution.replan_count += 1
        execution.plan = new_plan
        print_plan(new_plan)

    final = _synthesise(execution, client)
    execution.final_answer = final
    print(f"\nFINAL ANSWER:\n{final}")
    print(
        f"\nRUN STATS: "
        f"tool_calls={execution.total_tool_calls}  "
        f"replans={execution.replan_count}"
    )


# ----------------------------------------------------------------------------
# Demo 5 — max-step ceiling
# ----------------------------------------------------------------------------


def demo_5_max_step_ceiling() -> None:
    print_block(
        "DEMO 5 — max-step ceiling",
        "Hand-build a 30-step plan to demonstrate the executor's PROD PATTERN "
        "max_steps cap. The executor stops at the configured ceiling and "
        "records skipped StepResults for the rest.",
    )
    synthetic_goal = "(synthetic) run a long chain of cheap arithmetic steps"
    print(f"\nGOAL:\n  {synthetic_goal}\n")

    steps = []
    for i in range(1, 31):
        steps.append(Step(
            id=f"s{i}",
            description=f"Tiny calculator step {i}",
            tool="calculator",
            args={"expression": f"{i} + 1"},
        ))
    plan = Plan(
        goal=synthetic_goal,
        steps=steps,
        synthesis_hint="(none)",
    )

    executor = Executor(max_steps=12, on_step_done=print_step_event)
    execution = executor.run(plan)

    print(
        f"\nRUN STATS: planned={len(plan.steps)}  "
        f"executed={execution.total_tool_calls}  "
        f"skipped={sum(1 for r in execution.results if r.status == 'skipped')}"
    )
    print(
        "  The executor refused to run beyond 12 steps. This is the cheapest "
        "production circuit-breaker you can install in any agent."
    )


# ----------------------------------------------------------------------------
# Synthesis — small LLM call that composes the final answer.
# ----------------------------------------------------------------------------


SYNTH_SYSTEM_PROMPT = (
    "You are the synthesiser at the end of a Plan-and-Execute agent. "
    "You read the original GOAL, the PLAN, and the OBSERVATIONS that came "
    "back from each step. Write a short, direct answer to the goal — no "
    "preamble, no apology, no meta-commentary."
)


def _synthesise(execution: ExecutionResult, client: LLMClient) -> str:
    obs_lines = "\n".join(
        f"  {r.step_id}: {r.observation[:200]}"
        for r in execution.results if r.status != "skipped"
    )
    user = (
        f"GOAL: {execution.plan.goal}\n\n"
        f"PLAN SYNTHESIS HINT: {execution.plan.synthesis_hint}\n\n"
        f"OBSERVATIONS:\n{obs_lines}\n\n"
        "Synthesise the final answer now."
    )
    return client.complete(system=SYNTH_SYSTEM_PROMPT, user=user, max_tokens=300)


# ----------------------------------------------------------------------------
# CLI dispatcher
# ----------------------------------------------------------------------------


DEMOS = {
    "1": demo_1_react_baseline,
    "2": demo_2_plan_and_execute,
    "3": demo_3_side_by_side,
    "4": demo_4_drift_and_replan,
    "5": demo_5_max_step_ceiling,
}


def _usage() -> None:
    print(__doc__)
    print(f"Available tools in this build: {', '.join(tool_names())}")


def main(argv: list) -> int:
    if len(argv) < 2 or argv[1] not in DEMOS:
        _usage()
        return 0
    DEMOS[argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))