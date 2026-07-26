"""
Session XX — tools.py  (Phase 5 template)

Tool dispatcher pattern. Maps a tool name + JSON args to a Python
function. Single entry point so the trace logger sees every tool call
in one consistent shape.

Same dispatcher shape as S30 and S33. Add one entry to _REGISTRY for
each tool your agent can call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


# ---------------------------------------------------------------------
# ToolResult — what dispatch returns
# ---------------------------------------------------------------------


@dataclass
class ToolResult:
    tool_name: str
    ok: bool
    payload: Any
    error: str = ""

    def to_text(self) -> str:
        """Render the payload as the string the LLM will see."""

        if not self.ok:
            return f"ERROR ({self.tool_name}): {self.error}"
        # TODO — per-tool formatting. Default: dump JSON.
        return json.dumps(self.payload, default=str)


# ---------------------------------------------------------------------
# Per-tool dispatchers
# ---------------------------------------------------------------------


def _dispatch_TODO_tool(args: Dict[str, Any]) -> ToolResult:
    """TODO — implement this tool."""

    # Validate inputs.
    required = args.get("required_arg")
    if not isinstance(required, str) or not required.strip():
        return ToolResult(
            tool_name="TODO_tool",
            ok=False,
            payload=None,
            error="missing or empty 'required_arg'",
        )

    # Call the underlying function. Never raise — surface errors as text.
    try:
        # result = TODO_function(required)
        result = "TODO result"
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            tool_name="TODO_tool",
            ok=False,
            payload=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    return ToolResult(tool_name="TODO_tool", ok=True, payload=result)


# TODO — add one _dispatch_* function per tool.


# ---------------------------------------------------------------------
# Registry + main dispatcher
# ---------------------------------------------------------------------


_REGISTRY: Dict[str, Callable[[Dict[str, Any]], ToolResult]] = {
    "TODO_tool": _dispatch_TODO_tool,
    # TODO — register all your tools here.
}


def dispatch(tool_name: str, args: Dict[str, Any]) -> ToolResult:
    """Single entry point for every tool call. Never raises."""

    fn = _REGISTRY.get(tool_name)
    if fn is None:
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            payload=None,
            error=f"unknown tool '{tool_name}'",
        )
    return fn(args or {})


def known_tools() -> List[str]:
    return list(_REGISTRY.keys())
