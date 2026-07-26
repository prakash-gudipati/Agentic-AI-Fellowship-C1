"""
Session 36 — mcp_client.py

The MCP CLIENT. This is the other half of the contract: the side that
DISCOVERS and CALLS the tools a server advertises.

The official `mcp` SDK client API is async (it streams over stdio). For
teaching, we wrap that async machinery in a small SYNCHRONOUS facade so the
agent loop in agent.py can stay plain, top-to-bottom Python:

    client = MCPClient.launch_stdio("python", ["server.py"])
    tools = client.list_tools()          # discover
    result = client.call_tool("read_file", {"path": "notes.txt"})  # use
    client.close()

Under the hood we run one asyncio event loop on a background thread and keep a
single live ClientSession on it. Every public method submits a coroutine to
that loop and blocks for the result. This is a common production pattern when
a synchronous codebase needs to talk to an async library — worth seeing once.
"""

from __future__ import annotations

import threading
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class ToolSpec:
    """A tool as the server advertises it — name, description, input schema.

    This is the data an agent uses to decide WHICH tool to call. It is the
    MCP equivalent of the hand-written tool schema from Session 30 — except
    the agent builder did not write it. The SERVER published it.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPClient:
    """Synchronous facade over an async MCP stdio client session."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._cm_stack: List[Any] = []
        self._error: Optional[BaseException] = None

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def launch_stdio(cls, command: str, args: List[str]) -> "MCPClient":
        """Spawn `command args...` as an MCP server and connect over stdio."""

        client = cls()
        client._params = StdioServerParameters(command=command, args=args)
        client._thread = threading.Thread(target=client._run_loop, daemon=True)
        client._thread.start()
        client._ready.wait(timeout=30)
        if client._error is not None:
            raise client._error
        if client._session is None:
            raise RuntimeError("MCP client failed to initialise session")
        return client

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as exc:  # surfaced to launch_stdio()
            self._error = exc
            self._ready.set()

    async def _serve(self) -> None:
        # Open the transport + session, keep them open until _stop is set.
        async with stdio_client(self._params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                while not self._stop.is_set():
                    await asyncio.sleep(0.05)

    def close(self) -> None:
        """Shut the session and the server subprocess down cleanly."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- the two operations every MCP client needs -------------------------

    def list_tools(self) -> List[ToolSpec]:
        """Ask the server what tools it exposes (the discovery handshake)."""

        result = self._submit(self._session.list_tools())
        specs: List[ToolSpec] = []
        for t in result.tools:
            specs.append(
                ToolSpec(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema or {},
                )
            )
        return specs

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Invoke one tool by name and return its text result."""

        result = self._submit(self._session.call_tool(name, arguments))
        parts: List[str] = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(text)
        return "\n".join(parts) if parts else "(no text content returned)"

    # -- internal ----------------------------------------------------------

    def _submit(self, coro: Any) -> Any:
        if self._loop is None:
            raise RuntimeError("event loop not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)


def tools_as_anthropic_schema(specs: List[ToolSpec]) -> List[Dict[str, Any]]:
    """Convert MCP ToolSpecs into the tool format the Anthropic SDK expects.

    This single adapter is why the agent never hand-writes a schema again:
    whatever the server advertises becomes the agent's tool list automatically.
    """

    out: List[Dict[str, Any]] = []
    for s in specs:
        out.append(
            {
                "name": s.name,
                "description": s.description,
                "input_schema": s.input_schema or {"type": "object", "properties": {}},
            }
        )
    return out
