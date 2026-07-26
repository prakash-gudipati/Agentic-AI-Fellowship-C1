"""
Session 33 — tools.py

Tool dispatcher. Maps a tool name + JSON args to a Python function.

Why a dispatcher (and not direct calls from the agent loop)?
  - Single point of audit. Every tool invocation flows through dispatch()
    so the trace logger sees one consistent shape.
  - The agent's prompt doesn't need to know about Python — it works in
    JSON. The dispatcher is the JSON-to-Python boundary.
  - Adding a tool is a one-liner in the registry below. The agent loop
    never changes.

This is the same shape as S30's tool dispatcher, scaled down to two
tools because S33 is about retrieval discipline, not tool surface area.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from retrieval_tools import list_documents, search_kb


# ----------------------------------------------------------------------------
# ToolResult — what dispatch returns
# ----------------------------------------------------------------------------


@dataclass
class ToolResult:
    """The dispatcher returns one of these for every tool call.

    The agent loop turns it into a `tool_result` block for the LLM.
    """

    tool_name: str
    ok: bool
    payload: Any
    error: str = ""

    def to_text(self) -> str:
        """Render the payload as the string the LLM will see."""

        if not self.ok:
            return f"ERROR ({self.tool_name}): {self.error}"
        if self.tool_name == "search_kb":
            chunks = self.payload
            if not chunks:
                return "[search_kb] returned 0 chunks."
            lines = [f"[search_kb] {len(chunks)} chunks:"]
            for i, c in enumerate(chunks, 1):
                lines.append(
                    f"  ({i}) chunk_id={c.chunk_id}  source={c.source}\n"
                    f"      {c.text}"
                )
            return "\n".join(lines)
        if self.tool_name == "list_documents":
            docs = self.payload
            return "[list_documents]\n  - " + "\n  - ".join(docs)
        return json.dumps(self.payload, default=str)


# ----------------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------------


def _dispatch_search_kb(args: Dict[str, Any]) -> ToolResult:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return ToolResult(
            tool_name="search_kb",
            ok=False,
            payload=None,
            error="missing or empty 'query' argument",
        )
    k = int(args.get("k", 4))
    source_filter = args.get("source_filter") or None
    try:
        chunks = search_kb(
            query=query,
            k=k,
            source_filter=source_filter,
        )
    except Exception as exc:  # noqa: BLE001 — surface to the agent as text
        return ToolResult(
            tool_name="search_kb",
            ok=False,
            payload=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    return ToolResult(tool_name="search_kb", ok=True, payload=chunks)


def _dispatch_list_documents(args: Dict[str, Any]) -> ToolResult:
    try:
        docs = list_documents()
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            tool_name="list_documents",
            ok=False,
            payload=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    return ToolResult(tool_name="list_documents", ok=True, payload=docs)


_REGISTRY: Dict[str, Callable[[Dict[str, Any]], ToolResult]] = {
    "search_kb": _dispatch_search_kb,
    "list_documents": _dispatch_list_documents,
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
