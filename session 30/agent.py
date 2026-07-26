"""
Session 30 — agent.py

The function-calling agent. Same shape as the S29 ReAct loop, but no regex.

The new loop, in plain English:

    1.  Ask the provider for the next turn, handing it the conversation
        so far AND the tool schemas. The API may return text, OR a list
        of structured tool calls, OR both.

    2.  If the model returned no tool calls → it is done. Return the text.

    3.  Otherwise, for each tool call:
            - look up the tool in the registry
            - run it through the retry helper
            - record one ActionLog line (the new PROD PATTERN)
            - collect the result, keyed by the provider's call_id

    4.  Echo the assistant turn back into the conversation, then add a
        tool-result message carrying every result we collected.

    5.  Loop, up to MAX_TURNS.

What changed from S29:
  - No regex. No stop_sequences. The provider gives us structured calls.
  - Multiple tool calls per turn are supported transparently — the
    provider may emit two or three tool_use blocks in one response
    ("parallel tool calling"), and we run them in order before looping.
  - The loop is provider-agnostic. Swap AnthropicProvider for OpenAIProvider
    and the same questions run on a different LLM with zero loop changes.

Production patterns:
  - Agent action logging — one structured JSONL line per tool call    (S30 NEW)
  - Retry around every external call                                  (S29)
  - max_turns guard so the loop is provably bounded                   (S29)
  - Provider-agnostic interface                                       (S30 NEW)
  - try/except on the model call itself                               (S3)
  - Env var for API keys                                              (S8)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from action_log import ActionLog, STATUS_OK, STATUS_ERROR, new_run_id
from prompts import get_system_prompt
from providers.anthropic_provider import AnthropicProvider
from providers.base import (
    AssistantMessage,
    AssistantToolCall,
    Provider,
    ToolResultMessage,
)
from providers.openai_provider import OpenAIProvider
from retry import run_tool_with_retry
from tools import Tool, build_tool_lookup
import trace_logger as tl


load_dotenv()


# ── Defaults — tune in ONE place ────────────────────────────────────────────
MAX_TURNS = 8                              # hard ceiling — prevents runaway loops


@dataclass
class AgentRun:
    """Everything we recorded during one agent run. Useful for evals later."""
    question: str
    final_answer: Optional[str]
    turns_used: int
    stopped_reason: str
    run_id: str
    tool_call_count: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)


class ToolCallingAgent:
    """A minimal function-calling agent.

    Pure Python, native tool calling, one provider behind a small interface.
    No frameworks.
    """

    def __init__(
        self,
        *,
        tools: List[Tool],
        provider: Optional[Provider] = None,
        max_turns: int = MAX_TURNS,
        action_log_path: str = "action_log.jsonl",
        tool_choice: Any = None,
    ) -> None:
        if provider is None:
            provider = AnthropicProvider()
        self.provider = provider
        self.tools = list(tools)
        self.tool_lookup = build_tool_lookup(self.tools)
        self.tools_formatted = self.provider.format_tools(self.tools)
        self.system_prompt = get_system_prompt()
        self.max_turns = max_turns
        self.action_log_path = action_log_path
        # tool_choice steers tool selection:
        #   None / "auto" — model decides whether to use a tool (default)
        #   "any"         — model MUST call SOME tool
        #   {"name": X}   — model MUST call THIS specific tool
        # Only applied on the FIRST turn — after that we revert to auto so
        # the agent can answer once it has the data.
        self.tool_choice = tool_choice

    # ── public entrypoint ───────────────────────────────────────────────
    def run(self, question: str) -> AgentRun:
        """Solve a single question. Returns the run record."""
        run_id = new_run_id()
        action_log = ActionLog(run_id=run_id, log_path=self.action_log_path)

        tl.log_info(f"[provider={self.provider.name}] Question: {question}")

        # We carry the full conversation so the provider sees every prior
        # tool result. Format-wise this is a plain list of dicts — the
        # adapter handles role/content shape.
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": question},
        ]

        tool_call_count = 0

        for turn_number in range(1, self.max_turns + 1):
            tl.log_turn_header(turn_number)

            # 1. Ask the provider for the next turn.
            # tool_choice applies to the FIRST turn only. After that we let
            # the model decide whether to keep calling tools or to answer.
            tc_this_turn = self.tool_choice if turn_number == 1 else None
            try:
                assistant_message = self.provider.next_turn(
                    system_prompt=self.system_prompt,
                    messages=messages,
                    tools_formatted=self.tools_formatted,
                    tool_choice=tc_this_turn,
                )
            except Exception as exc:                            # noqa: BLE001
                tl.log_error(f"Provider call failed: {exc!r}")
                raise

            self._log_assistant_text(assistant_message)

            # 2. Echo the assistant turn back into the conversation.
            messages.append(self._assistant_message_dict(assistant_message))

            # 3. No tool calls → the model is done.
            if not assistant_message.tool_calls:
                final_answer = assistant_message.text or "(model returned no text)"
                tl.log_final_answer(final_answer)
                return AgentRun(
                    question=question,
                    final_answer=final_answer,
                    turns_used=turn_number,
                    stopped_reason="final_answer",
                    run_id=run_id,
                    tool_call_count=tool_call_count,
                    messages=messages,
                )

            # 4. Execute each tool call. Collect results keyed by call_id.
            tool_results = ToolResultMessage()
            for tool_call in assistant_message.tool_calls:
                tool_call_count += 1
                result = self._run_one_tool_call(
                    tool_call=tool_call,
                    turn_number=turn_number,
                    action_log=action_log,
                )
                tool_results.add(
                    call_id=tool_call.call_id,
                    content=result,
                )

            # 5. Hand the results back to the provider via the next message.
            next_message = self.provider.build_tool_result_message(tool_results)
            if "_messages" in next_message:
                # OpenAI flavour: one tool message per result. Extend.
                messages.extend(next_message["_messages"])
            else:
                messages.append(next_message)

        # Hit the max-turns guard.
        tl.log_error(f"Hit max_turns={self.max_turns} without a final answer.")
        return AgentRun(
            question=question,
            final_answer=None,
            turns_used=self.max_turns,
            stopped_reason="max_turns",
            run_id=run_id,
            tool_call_count=tool_call_count,
            messages=messages,
        )

    # ── one tool call, fully instrumented ───────────────────────────────
    def _run_one_tool_call(
        self,
        *,
        tool_call: AssistantToolCall,
        turn_number: int,
        action_log: ActionLog,
    ) -> str:
        """Execute one tool call with retry + structured logging."""
        tl.log_action(tool_call.tool_name, tool_call.tool_args)

        tool = self.tool_lookup.get(tool_call.tool_name)
        if tool is None:
            observation = (
                f"ERROR: no tool named '{tool_call.tool_name}'. "
                f"Available tools: {sorted(self.tool_lookup.keys())}"
            )
            # Still log the failed dispatch — silent gaps in the log are a bug.
            with action_log.measure(
                turn=turn_number,
                tool_name=tool_call.tool_name,
                tool_args=tool_call.tool_args,
            ) as ctx:
                ctx.finish(observation, attempts=0, status=STATUS_ERROR)
            tl.log_observation(observation)
            return observation

        # The action-log context manager measures duration and writes the line.
        with action_log.measure(
            turn=turn_number,
            tool_name=tool_call.tool_name,
            tool_args=tool_call.tool_args,
        ) as ctx:
            observation, attempts = run_tool_with_retry(tool.run, tool_call.tool_args)
            tl.log_retry(attempts, tool_call.tool_name)
            ctx.finish(observation, attempts=attempts)

        tl.log_observation(observation)
        return observation

    # ── small helpers ───────────────────────────────────────────────────
    @staticmethod
    def _log_assistant_text(message: AssistantMessage) -> None:
        if message.text:
            tl.log_thought(message.text)

    def _assistant_message_dict(self, message: AssistantMessage) -> Dict[str, Any]:
        """Provider-specific echo of the assistant turn."""
        if isinstance(self.provider, AnthropicProvider):
            return AnthropicProvider.assistant_message_from_response(message)
        if isinstance(self.provider, OpenAIProvider):
            return OpenAIProvider.assistant_message_from_response(message)
        raise RuntimeError(f"Unknown provider type: {type(self.provider).__name__}")
