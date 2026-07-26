"""
Session 36 — server.py

The MCP SERVER. This is the "tool contract" half of the session.

We use FastMCP, the server class that ships INSIDE the official `mcp` Python
SDK (`from mcp.server.fastmcp import FastMCP`). FastMCP turns a plain Python
function into a standardised MCP tool: it reads the function's type hints to
build the input schema and its docstring to build the description. That
description and schema are exactly what any MCP CLIENT sees when it asks the
server "what can you do?".

Three tools, as in the curriculum:
    read_file   — read a text file from the sandboxed workspace
    write_file  — create / overwrite a text file in the sandboxed workspace
    search_web  — offline canned web search

Run this server directly to serve over stdio (the transport for LOCAL servers
the user controls):

    python server.py

A client (mcp_client.py, agent.py, or Claude Desktop) launches this file as a
subprocess and talks to it over stdin/stdout. To serve the SAME tools to a
REMOTE client you change ONE line — `mcp.run(transport="streamable-http")` —
and nothing else. That transport-independence is the universal-adapter payoff.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

import tools_impl


# The server's name is part of its identity to clients.
mcp = FastMCP("fellowship-tools")


@mcp.tool()
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the workspace.

    Args:
        path: File name or relative path inside the workspace, e.g. "notes.txt".
    """

    return tools_impl.read_file_impl(path)


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Create or overwrite a UTF-8 text file in the workspace.

    Args:
        path: File name or relative path inside the workspace.
        content: The full text to write to the file.
    """

    return tools_impl.write_file_impl(path, content)


@mcp.tool()
def search_web(query: str, max_results: int = 3) -> str:
    """Search the web for a query and return the top ranked snippets.

    Args:
        query: The search query in plain English.
        max_results: How many results to return (default 3).
    """

    return tools_impl.search_web_impl(query, max_results)


if __name__ == "__main__":
    # stdio is the default transport: the client owns this process and reads
    # our stdout. For a remote deployment you would instead call:
    #     mcp.run(transport="streamable-http")
    mcp.run()
