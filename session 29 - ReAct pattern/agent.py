"""
Session 29 — agent.py

The ReAct loop. The heart of every agent in this course.

Why does this exist as its own module?
  - This file does ONE thing: drive the Thought → Action → Observation
    loop. Tools, prompt, parsing, logging, retry — all of those are
    delegated to their own files. The loop file should fit on a
    single screen for a reason. The agent loop is the concept; the rest
    is plumbing.

The loop (one turn):

    1. Call the LLM with the running conversation, stop=["Observation:"]
    2. Parse the model's reply:
         - Final Answer? → return it
         - Action?       → execute the tool (with retry)
         - Neither?      → break with a "model went off-format" error
    3. Append "Observation: <result>" to the conversation
    4. Repeat, up to MAX_TURNS

Why stop=["Observation:"]:
  - The model writes "Thought: ... Action: tool[x] Observation:" and STOPS
    right before Observation. We then execute the tool and write the REAL
    observation. This single trick is what makes the ReAct loop reliable.

Production patterns (S29):
  - Retry logic for tool calls                       (S29 NEW — the headline pattern)
  - Max-turns guard so the loop is provably bounded  (S29 NEW)
  - Stop-sequence contract for clean tool dispatch   (S29 NEW)
  - try/except around the LLM call itself            (S3)
  - Env var for API key                              (S8)
  - Structured trace logging                         (S14)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from anthropic import Anthropic
from dotenv import load_dotenv

from parser import parse_action, parse_final_answer
from prompts import build_system_prompt
from retry import run_tool_with_retry
from tools import TOOL_REGISTRY
import trace_logger as tl


load_dotenv()

# ── Defaults — tune in ONE place ───────────────────────────────────────────
MODEL_NAME = "claude-haiku-4-5-20251001"   # cheap + fast for the live demo
MAX_TURNS = 8                              # hard ceiling — prevents runaway loops
MAX_TOKENS_PER_TURN = 800
STOP_SEQUENCE = "Observation:"             # the LLM stops here on its own


@dataclass
class AgentRun:
    """Everything we recorded during one agent run. Useful for evals later."""
    question: str
    final_answer: str | None
    turns_used: int
    stopped_reason: str
    trace: List[str] = field(default_factory=list)


class ReactAgent:
    """A minimal ReAct agent. Pure Python + Anthropic SDK. No frameworks."""

    def __init__(self, model_name: str = MODEL_NAME, max_turns: int = MAX_TURNS):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
        self.max_turns = max_turns
        self.system_prompt = build_system_prompt()

    # ── public entrypoint ───────────────────────────────────────────────
    def run(self, question: str) -> AgentRun:
        """Solve a single question. Returns the run record."""
        tl.log_info(f"Question: {question}")

        conversation_so_far = f"Question: {question}\n"
        trace: List[str] = [conversation_so_far]

        for turn_number in range(1, self.max_turns + 1):
            tl.log_turn_header(turn_number)

            # 1. Ask the LLM what to do next
            model_reply = self._call_llm(conversation_so_far)
            trace.append(model_reply)
            self._log_reply(model_reply)

            # 2. Did it answer?
            answer = parse_final_answer(model_reply)
            if answer is not None:
                tl.log_final_answer(answer)
                return AgentRun(
                    question=question,
                    final_answer=answer,
                    turns_used=turn_number,
                    stopped_reason="final_answer",
                    trace=trace,
                )

            # 3. Did it ask for a tool?
            action = parse_action(model_reply)
            if action is None:
                tl.log_error(
                    "Model emitted neither Final Answer nor Action — stopping."
                )
                return AgentRun(
                    question=question,
                    final_answer=None,
                    turns_used=turn_number,
                    stopped_reason="off_format",
                    trace=trace,
                )

            tool = TOOL_REGISTRY.get(action.tool_name)
            if tool is None:
                observation = (
                    f"ERROR: no tool named '{action.tool_name}'. "
                    f"Available tools: {sorted(TOOL_REGISTRY.keys())}"
                )
            else:
                observation = run_tool_with_retry(
                    tool.run,
                    action.tool_input,
                    on_attempt=tl.log_tool_retry,
                )

            tl.log_observation(observation)

            # 4. Append the real observation and loop
            conversation_so_far += model_reply + f"\nObservation: {observation}\n"
            trace.append(f"Observation: {observation}")

        tl.log_error(f"Hit max_turns={self.max_turns} without a final answer.")
        return AgentRun(
            question=question,
            final_answer=None,
            turns_used=self.max_turns,
            stopped_reason="max_turns",
            trace=trace,
        )

    # ── internals ───────────────────────────────────────────────────────
    def _call_llm(self, conversation_so_far: str) -> str:
        """One LLM turn. Stops cleanly at 'Observation:'."""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=MAX_TOKENS_PER_TURN,
                system=self.system_prompt,
                stop_sequences=[STOP_SEQUENCE],
                messages=[{"role": "user", "content": conversation_so_far}],
            )
        except Exception as exc:                            # noqa: BLE001
            tl.log_error(f"LLM call failed: {exc!r}")
            raise

        # Anthropic returns a list of content blocks — for plain text we
        # just concatenate the .text fields.
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()

    @staticmethod
    def _log_reply(model_reply: str) -> None:
        """Pretty-print whichever Thought/Action/etc lines the model emitted."""
        for line in model_reply.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("thought:"):
                tl.log_thought(stripped[len("Thought:"):].strip())
            elif stripped.lower().startswith("action:"):
                # The parser handles structure; here we just echo the raw line.
                tl.log_action("(action)", stripped[len("Action:"):].strip())
            elif stripped.lower().startswith("final answer:"):
                pass  # logged by the caller after parse
            elif stripped:
                # any free-form line the model emitted (rare, but informative)
                sys_line = stripped
                tl.log_info(sys_line)
