"""
Session 30 - demo.py

Five demos:
  Demo 1: VAGUE schemas across 3 questions (Anthropic)
  Demo 2: SHARP schemas across the same 3 questions
  Demo 3: Multi-step research run on Anthropic
  Demo 4: Same multi-step question on OpenAI
  Demo 5: tool_choice auto vs any vs forced

Run it with:
    python demo.py            # runs all five
    python demo.py 2          # only Demo 2
    python demo.py 1 2 5      # selected demos in order

After every run, look at action_log.jsonl - the structured JSONL audit trail.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from action_log import summarise_run
from agent import ToolCallingAgent
from providers.anthropic_provider import AnthropicProvider
from providers.openai_provider import OpenAIProvider
from schema_quality import SHARP_TOOLS, VAGUE_TOOLS


SCHEMA_QUALITY_QUESTIONS: List[str] = [
    "What is 12 times 7?",
    "What is the GDP of India in 2024?",
    "Who was Alan Turing?",
]


MULTI_STEP_QUESTION = (
    "What is the GDP of India in 2024 divided by its population in 2024? "
    "Give the answer in dollars per person, rounded to the nearest dollar."
)


LOG_PATH = Path("action_log.jsonl")


def _banner(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def _print_log_summary(run_id: str) -> None:
    summary = summarise_run(LOG_PATH, run_id=run_id)
    print("\nAction log summary (this run only):")
    print(f"  run_id           : {run_id}")
    print(f"  tool_calls       : {summary['tool_calls']}")
    print(f"  errors           : {summary['errors']}")
    print(f"  schema_violations: {summary['schema_violations']}")
    print(f"  total_duration_ms: {summary['total_duration_ms']}")
    print(f"  by_tool          : {summary['by_tool']}")


def demo_1_vague_schemas() -> None:
    _banner('DEMO 1 - VAGUE schemas + forced tool use (tool_a / tool_b / tool_c, all "Returns a result."). Watch it miss.')
    # tool_choice="any" forces the model to call SOME tool on every question.
    # With three identical descriptions, every selection is a coin flip.
    # This is what production looks like when bad metadata meets a grounding requirement.
    agent = ToolCallingAgent(
        tools=VAGUE_TOOLS,
        provider=AnthropicProvider(),
        action_log_path=str(LOG_PATH),
        tool_choice="any",
    )
    for question in SCHEMA_QUALITY_QUESTIONS:
        print(f"\n--- Question: {question} ---")
        result = agent.run(question)
        _print_log_summary(result.run_id)


def demo_2_sharp_schemas() -> None:
    _banner("DEMO 2 - SHARP schemas. Same code, sharper words. Watch it hit.")
    agent = ToolCallingAgent(
        tools=SHARP_TOOLS,
        provider=AnthropicProvider(),
        action_log_path=str(LOG_PATH),
    )
    for question in SCHEMA_QUALITY_QUESTIONS:
        print(f"\n--- Question: {question} ---")
        result = agent.run(question)
        _print_log_summary(result.run_id)


def demo_3_multi_step_anthropic() -> None:
    _banner("DEMO 3 - Multi-step research (Anthropic). web_search x2 + calculator.")
    agent = ToolCallingAgent(
        tools=SHARP_TOOLS,
        provider=AnthropicProvider(),
        action_log_path=str(LOG_PATH),
    )
    result = agent.run(MULTI_STEP_QUESTION)
    _print_log_summary(result.run_id)


def demo_4_multi_step_openai() -> None:
    _banner("DEMO 4 - Same question, OpenAI provider. Same agent code.")
    agent = ToolCallingAgent(
        tools=SHARP_TOOLS,
        provider=OpenAIProvider(),
        action_log_path=str(LOG_PATH),
    )
    result = agent.run(MULTI_STEP_QUESTION)
    _print_log_summary(result.run_id)


def demo_5_tool_choice_modes() -> None:
    _banner("DEMO 5 - tool_choice: auto vs any vs forced specific tool")
    question = "What is 12 times 7?"

    print("\n--- Mode 1: tool_choice=auto (model decides) ---")
    a = ToolCallingAgent(tools=SHARP_TOOLS, provider=AnthropicProvider(),
                         action_log_path=str(LOG_PATH), tool_choice=None)
    r = a.run(question)
    print(f"  tool_call_count={r.tool_call_count}  (often 0 - answered from memory)")
    _print_log_summary(r.run_id)

    print("\n--- Mode 2: tool_choice='any' (must call SOME tool) ---")
    a = ToolCallingAgent(tools=SHARP_TOOLS, provider=AnthropicProvider(),
                         action_log_path=str(LOG_PATH), tool_choice="any")
    r = a.run(question)
    print(f"  tool_call_count={r.tool_call_count}  (>=1 - model forced to ground)")
    _print_log_summary(r.run_id)

    print("\n--- Mode 3: tool_choice={'name':'calculator'} (force calculator) ---")
    a = ToolCallingAgent(tools=SHARP_TOOLS, provider=AnthropicProvider(),
                         action_log_path=str(LOG_PATH),
                         tool_choice={"name": "calculator"})
    r = a.run(question)
    print(f"  tool_call_count={r.tool_call_count}  (=1 - calculator called by force)")
    _print_log_summary(r.run_id)


DEMOS = {
    1: demo_1_vague_schemas,
    2: demo_2_sharp_schemas,
    3: demo_3_multi_step_anthropic,
    4: demo_4_multi_step_openai,
    5: demo_5_tool_choice_modes,
}


def main() -> None:
    args = sys.argv[1:]
    demo_ids = [int(a) for a in args] if args else list(DEMOS.keys())
    for demo_id in demo_ids:
        DEMOS[demo_id]()
    print("\nAll done. Tail action_log.jsonl to see every tool call ever made.")


if __name__ == "__main__":
    main()
