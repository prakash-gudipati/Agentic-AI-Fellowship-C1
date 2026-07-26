"""
Session 36 — tools_impl.py

The PURE tool logic, kept deliberately separate from the MCP server that
exposes it (server.py). Two reasons this split matters in production:

  1. The same functions can be unit-tested without standing up a server.
  2. The MCP layer in server.py stays a thin "contract" — it only declares
     the tool's name, description, and input schema, then delegates here.
     That is the whole point of MCP: the TOOL is one thing, the PROTOCOL
     that advertises it is another.

All file access is sandboxed to a single workspace directory so a misbehaving
agent cannot read or write outside it. search_web is OFFLINE — it answers from
a small canned corpus so every demo runs without a network connection or an
API key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Sandbox root for read_file / write_file.
# Override with S36_WORKSPACE_DIR for restrictive / networked filesystems.
# ---------------------------------------------------------------------------

_DEFAULT_WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE = Path(os.environ.get("S36_WORKSPACE_DIR") or _DEFAULT_WORKSPACE)


def _safe_path(relative_name: str) -> Path:
    """Resolve `relative_name` INSIDE the workspace and refuse to escape it.

    This is the production-grade guard: an agent that asks for
    '../../etc/passwd' gets a clean error, not a path traversal.
    """

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    candidate = (WORKSPACE / relative_name).resolve()
    workspace_root = WORKSPACE.resolve()
    if workspace_root not in candidate.parents and candidate != workspace_root:
        raise ValueError(
            f"refused: '{relative_name}' resolves outside the workspace sandbox"
        )
    return candidate


def read_file_impl(path: str) -> str:
    """Return the text contents of a file inside the workspace sandbox."""

    target = _safe_path(path)
    if not target.exists():
        return f"ERROR: file not found: {path}"
    if not target.is_file():
        return f"ERROR: not a file: {path}"
    return target.read_text(encoding="utf-8")


def write_file_impl(path: str, content: str) -> str:
    """Create or overwrite a file inside the workspace sandbox."""

    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} characters to {path}"


# ---------------------------------------------------------------------------
# Offline "web search" — a tiny canned corpus so the demos never touch the
# network. In a real server this would call a search API; the CONTRACT the
# agent sees (a query in, ranked snippets out) is identical either way.
# ---------------------------------------------------------------------------

_CANNED_WEB: List[dict] = [
    {
        "keywords": ["mcp", "model context protocol", "protocol"],
        "title": "Model Context Protocol — Introduction",
        "snippet": (
            "The Model Context Protocol (MCP) is an open standard that lets AI "
            "applications connect to tools and data sources through one common "
            "interface, instead of a custom integration per tool."
        ),
        "url": "https://modelcontextprotocol.io/introduction",
    },
    {
        "keywords": ["transport", "stdio", "http", "streamable"],
        "title": "MCP Transports — stdio and Streamable HTTP",
        "snippet": (
            "MCP defines two standard transports: stdio for local servers the "
            "user controls, and Streamable HTTP for remote servers. The older "
            "HTTP+SSE transport was deprecated in March 2025."
        ),
        "url": "https://modelcontextprotocol.io/docs/concepts/transports",
    },
    {
        "keywords": ["a2a", "agent-to-agent", "agent to agent"],
        "title": "A2A — Agent-to-Agent Protocol",
        "snippet": (
            "A2A is an open protocol for agents on different frameworks to "
            "communicate and delegate. Where MCP connects an agent to TOOLS, "
            "A2A connects an agent to OTHER AGENTS."
        ),
        "url": "https://a2aproject.github.io/A2A/",
    },
    {
        "keywords": ["ag-ui", "agent user interaction", "user interface"],
        "title": "AG-UI — Agent-User Interaction Protocol",
        "snippet": (
            "AG-UI standardises how an agent streams updates to a user-facing "
            "application, so agents can escape plain-text chat into interactive "
            "UIs."
        ),
        "url": "https://docs.ag-ui.com/",
    },
]


def search_web_impl(query: str, max_results: int = 3) -> str:
    """Return ranked snippets for a query from an OFFLINE canned corpus.

    Ranking is a simple keyword-overlap score — enough to make the demo
    deterministic. The shape of the output (a numbered list of title /
    snippet / url) is what the agent depends on, not the ranking quality.
    """

    q_words = {w for w in query.lower().replace("?", " ").split() if len(w) > 2}
    scored = []
    for entry in _CANNED_WEB:
        hay = " ".join(entry["keywords"]) + " " + entry["title"].lower()
        score = sum(1 for kw in entry["keywords"] if kw in query.lower())
        score += sum(1 for w in q_words if w in hay)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    hits = [e for _, e in scored[: max(1, max_results)]]
    if not hits:
        return f"No results for: {query}"
    lines = [f"Top results for: {query}"]
    for i, e in enumerate(hits, start=1):
        lines.append(f"{i}. {e['title']}\n   {e['snippet']}\n   {e['url']}")
    return "\n".join(lines)
