"""
Session 35 — tools.py

Single tool dispatcher placeholder. Most S35 agents don't use external
tools — the multi-agent value is in coordination, not new tool design
(S30 was the tool-design session). This file is kept so the structure
matches S34 and to give the bus_validation demo something to call.
"""

from __future__ import annotations

from typing import Any, Dict


def dispatch(name: str, args: Dict[str, Any]) -> str:
    """Dispatch a tool call. Never raises — always returns a string
    observation (PROD PATTERN reused from S30).
    """

    if name == "noop":
        return "ok"
    if name == "echo":
        return str(args.get("text", ""))
    return f"(tools.py) unknown tool: {name}"
