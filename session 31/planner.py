"""
Session 31 — planner.py

The Planner converts a free-text GOAL into a typed PLAN.

Pipeline:
  1. Render the system prompt — describes the contract the planner must
     respect (return JSON only, use only known tools, declare dependencies).
  2. Ask the LLM for a plan.
  3. Parse the JSON. If the model wrapped it in markdown fences or chatter,
     extract the first {...} block.
  4. Build a Plan object out of the dict.
  5. Validate the plan against the executor's allow-list of tools.
  6. Return the Plan, or raise PlanValidationError so the caller can
     decide whether to retry the LLM call or surface the error.

There are three failure modes worth knowing by name:
  - JSON ERROR        — the LLM produced un-parseable JSON. Rare, but
                        survivable: ask once more with a tighter prompt.
  - INVALID PLAN      — JSON parsed but references an unknown tool, has
                        a cycle, or duplicate ids. Raises
                        PlanValidationError; caller can either retry the
                        planner or bail with a clear error to the user.
  - HALLUCINATED ARGS — the JSON looks fine, but a calculator step has
                        a non-arithmetic string in its expression. We do
                        NOT catch this here — the executor runs the step,
                        the tool returns "ERROR: ...", and the monitor
                        catches it. That is the whole point of the
                        monitor / replan loop.

Production patterns active in this module:
  - PLAN AS ARTIFACT — see plan_types.py. The planner returns a typed
                       object, not free text.
  - VALIDATE-AT-BOUNDARY — every untrusted input (LLM output) is parsed
                       and validated before any executor code touches it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from llm_client import LLMClient
from plan_types import Plan, PlanValidationError, Step, validate_plan
from tools import render_catalog, tool_names


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


PLAN_SYSTEM_PROMPT = """You are a careful planning assistant.

You receive a USER GOAL and a CATALOG of tools. Your job is to PRODUCE A PLAN
AS JSON. The plan is a sequence of small steps that, when executed in order,
will let a downstream agent answer the goal.

Hard rules:
  1. Output a SINGLE JSON object. No prose before or after. No markdown
     fences. Just the JSON.
  2. Every step's "tool" field must be one of the tool names in the catalog.
     Do not invent new tools.
  3. Use the args shape the tool's description specifies.
  4. Steps that depend on the OBSERVATION of an earlier step must list that
     step's id in "depends_on". The executor walks the plan in dependency
     order.
  5. Keep the plan SHORT. Each step should do ONE thing. If a step needs
     a number that another step produced, prefer to compute that number
     in a separate calculator step rather than embedding it inline.
  6. Always end with one "synthesis_hint" — a one-line instruction telling
     the downstream synthesiser how to combine the observations into the
     final answer.

JSON shape:
  {
    "synthesis_hint": "<one-line instruction for the final answer>",
    "steps": [
      {
        "id": "s1",
        "description": "<human-readable line>",
        "tool": "<tool name>",
        "args": { ... },
        "depends_on": []
      },
      ...
    ]
  }
"""


def make_plan(
    goal: str,
    client: Optional[LLMClient] = None,
    max_tokens: int = 1024,
) -> Plan:
    """
    Produce and validate a Plan for the given goal.

    Raises:
      PlanValidationError — JSON parsed but plan is malformed.
      ValueError          — LLM returned no parseable JSON.
    """

    client = client or LLMClient()

    user = (
        f"CATALOG OF TOOLS:\n{render_catalog()}\n\n"
        f"USER GOAL: {goal}\n\n"
        "Produce a plan as JSON now."
    )

    raw = client.complete(system=PLAN_SYSTEM_PROMPT, user=user,
                          max_tokens=max_tokens)

    plan_dict = _extract_json(raw)
    plan = _dict_to_plan(plan_dict, goal)
    validate_plan(plan, allowed_tools=tool_names())
    return plan


# ----------------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Pull the first {...} block out of a possibly-chatty LLM response.

    Modern Claude usually returns clean JSON when asked. This helper is
    defensive cover for two cases the instructor will sometimes see live:
      - the model wrapped the JSON in ```json ... ``` fences
      - the model prefixed it with a sentence like "Here is the plan:"
    """

    raw = raw.strip()
    if not raw:
        raise ValueError("planner LLM returned an empty response")

    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise ValueError(
            f"planner LLM did not return JSON. Got: {raw[:200]!r}"
        )

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"planner LLM JSON failed to parse: {exc}. "
            f"Raw response: {raw[:200]!r}"
        ) from exc


def _dict_to_plan(plan_dict: Dict[str, Any], goal: str) -> Plan:
    """Turn a parsed JSON dict into a Plan object."""

    if not isinstance(plan_dict, dict):
        raise PlanValidationError(
            f"plan must be a JSON object, got {type(plan_dict).__name__}"
        )

    steps_in = plan_dict.get("steps")
    if not isinstance(steps_in, list):
        raise PlanValidationError("plan missing 'steps' list")

    steps: List[Step] = []
    for raw_step in steps_in:
        if not isinstance(raw_step, dict):
            raise PlanValidationError(
                f"step entries must be objects, got {type(raw_step).__name__}"
            )
        try:
            steps.append(
                Step(
                    id=str(raw_step["id"]),
                    description=str(raw_step.get("description", "")),
                    tool=str(raw_step["tool"]),
                    args=dict(raw_step.get("args", {}) or {}),
                    depends_on=list(raw_step.get("depends_on", []) or []),
                )
            )
        except KeyError as exc:
            raise PlanValidationError(
                f"step entry is missing required field {exc.args[0]!r}"
            ) from exc

    return Plan(
        goal=goal,
        steps=steps,
        synthesis_hint=str(plan_dict.get("synthesis_hint", "")),
    )


# ----------------------------------------------------------------------------
# CLI smoke test
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    # Run with FAKE_LLM=1 to see a canned plan without an API key.
    demo_goal = (
        "Compare India and United States per-capita GDP in 2024 and "
        "tell me which is larger plus the ratio."
    )
    plan = make_plan(demo_goal)
    print(f"GOAL: {plan.goal}")
    print(f"SYNTHESIS HINT: {plan.synthesis_hint}")
    print("STEPS:")
    for s in plan.steps:
        deps = f" (after {','.join(s.depends_on)})" if s.depends_on else ""
        print(f"  {s.id}{deps}: {s.description}")
        print(f"      tool={s.tool} args={s.args}")
