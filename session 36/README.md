# Session 36 — MCP — Model Context Protocol (Reference Code)

Portfolio session. You build an **MCP server** that exposes tools, and wire
your Phase 5 **agent** in as an **MCP client** that discovers and calls them.

> **The one idea:** the tool and the protocol that advertises it are two
> separate things. Write a tool once behind an MCP server, and *any* client —
> your agent, a raw script, Claude Desktop — can use it. Universal adapter.

## What the demos prove

| Demo | What it shows | LLM? |
|------|---------------|------|
| 1 | A raw MCP client does the **discovery handshake** (`list_tools`) and calls a tool directly | no |
| 2 | The agent, wired as an MCP client, **reads a file over MCP** to answer a question | yes (faked) |
| 3 | The agent **writes then reads back** a file — multi-step tool use over MCP | yes (faked) |
| 4 | The agent uses **`search_web`** over MCP (offline corpus) | yes (faked) |
| 5 | A **second, separate client** uses the **same server**; plus the MCP / A2A / AG-UI landscape | no |

The **MCP transport is always real** — a server subprocess is launched and
spoken to over stdio. Only the agent's *choice of tool* is faked when
`FAKE_LLM=1`, so every demo runs offline with no API key.

## Run it

```bash
pip install -r requirements.txt   # or: pip install "mcp>=1.2.0"

# Recommended Phase 5 smoke-test (offline, no key):
PYTHONPYCACHEPREFIX=/tmp/s36_pycache \
S36_WORKSPACE_DIR=/tmp/s36_workspace \
FAKE_LLM=1 python demo.py 1
```

Swap `FAKE_LLM=1` for a real `ANTHROPIC_API_KEY` in `.env` to use a live brain.
The server never needs a key.

Run the server on its own (it speaks stdio):

```bash
python server.py
```

## Files

```
server.py             FastMCP server — read_file, write_file, search_web (the tool contract)
tools_impl.py         Pure tool logic (sandboxed file I/O + offline search), unit-testable
mcp_client.py         Synchronous facade over the async MCP stdio client + ToolSpec discovery
agent.py              MCPAgent — discovers MCP tools, runs the S29/S30 decide→act→observe loop
llm_client.py         Anthropic wrapper + FAKE_LLM offline agent brain
prompts.py            Agent system prompt (opener-phrase rule)
protocol_landscape.py MCP vs A2A vs AG-UI reference (printed in Demo 5)
trace_logger.py       ANSI trace printers (NO_COLOR=1 to disable)
demo.py               5 demos
workspace/            sandbox for read_file / write_file (seeded with product_brief.txt)
```

## PROD PATTERNS in this code

- **MCP Server as a Tool Contract** — `server.py` declares each tool's name,
  schema (from type hints) and description (from the docstring). That is the
  entire public surface. Clients depend on the contract, not the code.
- **Agent-as-MCP-Client** — `agent.py` never hard-codes a tool. It calls
  `list_tools()` at startup and turns the result into its tool list
  (`tools_as_anthropic_schema`). New server tool → agent gets it for free.
- **Sandboxed tool execution** — `tools_impl._safe_path` refuses any path that
  escapes the workspace. A tool exposed to an agent must be safe by construction.
- **Transport independence** — stdio today; one line (`transport="streamable-http"`)
  serves the same tools remotely. The deprecated HTTP+SSE transport is not used.

## Verify each pattern in the trace

- `DISCOVER: 3 tools from MCP server` (Demos 2-4) — the agent learned its tools
  from the server, not from its own source.
- `TOOL CALL → read_file ... [over MCP]` — the call left the agent process and
  travelled to the server over the protocol.
- Demo 5's "second client discovered the same 3 tools" — one contract, many clients.
