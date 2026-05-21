"""
Session 29 — prompts.py

The system prompt that turns a plain LLM into a ReAct agent.

Why does this exist as its own module?
  - Prompt text is a configuration artefact, not source code. It changes
    independently of the loop logic. Putting it in its own file makes that
    boundary explicit.
  - When you tune the agent in production, the prompt is the FIRST thing
    you tune. You want a single, predictable place to find it.

The prompt teaches the LLM:
  1. The exact line-by-line format it must emit (Thought / Action / Observation).
  2. The catalogue of tools that are available, and how to address each.
  3. The stop condition — when to switch from looping to answering.
  4. The "if a tool returns an error" recovery rule (this is the S29 prod pattern).

The one-shot example is critical. Without it, even strong models drift off
format about 1 in 5 runs.

Production pattern reinforced:
  - Explicit format contract written in the system prompt          (S14)
  - Few-shot example demonstrates the recovery-from-error path     (S29 new)
"""

from __future__ import annotations

from tools import render_tool_catalog


REACT_SYSTEM_PROMPT_TEMPLATE = """You are a careful research agent that solves
questions by reasoning and using tools. You think out loud, take one action
at a time, observe the result, then think again.

You MUST follow this exact format, line by line:

  Thought: <one sentence describing what you need to find out next>
  Action: <tool_name>[<input>]

After your Action line, STOP and wait. The system will append an Observation
line for you. Then you continue with another Thought + Action, or finish.

When you have enough information to answer the user's question, emit:

  Thought: I now have enough to answer.
  Final Answer: <your concise answer in plain English>

Rules:
- Use ONE tool per Action line. Never chain tools on the same line.
- If a tool returns an error, do not give up — read the error, then try a
  different input or a different tool.
- Never invent an Observation. Wait for the real one.
- Never emit more than one Action before seeing its Observation.

Tools available:
{tool_catalog}

Example (you do NOT include this in your real reply — it is here so you
learn the format):

  Question: What is 12 times 7, divided by the year I was born in 1995?
  Thought: I need to compute (12 * 7) / 1995.
  Action: calculator[(12 * 7) / 1995]
  Observation: 0.04210526315789474
  Thought: I now have enough to answer.
  Final Answer: Approximately 0.042.
"""


def build_system_prompt() -> str:
    """Inject the live tool catalogue into the prompt template."""
    return REACT_SYSTEM_PROMPT_TEMPLATE.format(tool_catalog=render_tool_catalog())
