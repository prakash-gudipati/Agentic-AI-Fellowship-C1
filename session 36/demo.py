"""
Session 36 — demo.py

CLI dispatcher for the MCP walkthrough demos.

    python demo.py 1   # Raw MCP client — discovery handshake + a direct call
    python demo.py 2   # Agent reads a file over MCP to answer a question
    python demo.py 3   # Agent writes a file over MCP, then reads it back
    python demo.py 4   # Agent uses search_web over MCP
    python demo.py 5   # Same server, a second client — the tool contract reused

The MCP transport is ALWAYS real (a server subprocess is launched and spoken to
over stdio). Only the agent's choice of tool is faked when FAKE_LLM=1.

PHASE 5 recommended smoke-test command:

    PYTHONPYCACHEPREFIX=/tmp/s36_pycache \\
    S36_WORKSPACE_DIR=/tmp/s36_workspace \\
    FAKE_LLM=1 python demo.py 1
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_HERE = Path(__file__).parent
_load_dotenv(_HERE / ".env")

from agent import MCPAgent  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from mcp_client import MCPClient  # noqa: E402
from protocol_landscape import print_landscape  # noqa: E402
from trace_logger import (  # noqa: E402
    handle_event,
    print_final_answer,
    print_section,
    print_tool_catalog,
    print_user_question,
)


if os.environ.get("FAKE_LLM", "") != "1" and not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit(
        "\nERROR: ANTHROPIC_API_KEY is not set.\n"
        "  Either export ANTHROPIC_API_KEY=sk-ant-...  (real agent brain)\n"
        "  or     export FAKE_LLM=1                    (offline agent brain)\n"
        "  The MCP server itself never needs a key.\n"
    )


# Server launch parameters — used by every demo. We run THIS Python with the
# server.py script; stdio_client spawns it as a subprocess.
_SERVER_CMD = sys.executable
_SERVER_ARGS = [str(_HERE / "server.py")]


def _new_client() -> MCPClient:
    return MCPClient.launch_stdio(_SERVER_CMD, _SERVER_ARGS)


def _cost_note(llm_calls: int) -> None:
    # Claude Haiku 4.5: $1 / 1M input, $5 / 1M output. ~600 in + 150 out / call.
    per_call = (600 * 1 + 150 * 5) / 1_000_000  # ≈ $0.00135
    usd = llm_calls * per_call
    inr = usd * 83
    print(f"COST: ~${usd:.4f} / run · ~Rs.{inr:.2f} / run  ({llm_calls} LLM calls; MCP transport is free)")


# ---------------------------------------------------------------------------
# Demo 1 — Raw MCP client: discovery handshake + one direct call (NO LLM)
# ---------------------------------------------------------------------------


def demo_1() -> None:
    print_section("Demo 1 — Raw MCP client (discovery + direct call)")
    print("  No agent, no LLM. Just a client talking to the server over stdio.")
    client = _new_client()
    try:
        specs = client.list_tools()
        print_tool_catalog(specs)
        print("\n  Calling read_file('product_brief.txt') directly:")
        result = client.call_tool("read_file", {"path": "product_brief.txt"})
        print_final_answer(result)
    finally:
        client.close()
    print("\n  Takeaway: the client never hard-coded the tools — it ASKED.")


# ---------------------------------------------------------------------------
# Demo 2 — Agent reads a file over MCP to answer a question
# ---------------------------------------------------------------------------


def demo_2() -> None:
    print_section("Demo 2 — Agent-as-MCP-client (read a file to answer)")
    client = _new_client()
    try:
        agent = MCPAgent(client, llm=LLMClient(), on_event=handle_event)
        q = "What is the launch date in product_brief.txt? Read the file first."
        print_user_question(q)
        answer = agent.answer(q)
        print_final_answer(answer)
        _cost_note(agent.llm.llm_calls)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Demo 3 — Agent writes a file over MCP, then reads it back
# ---------------------------------------------------------------------------


def demo_3() -> None:
    print_section("Demo 3 — Agent writes then reads back (multi-step over MCP)")
    client = _new_client()
    try:
        agent = MCPAgent(client, llm=LLMClient(), on_event=handle_event)
        q = (
            "Create a file called note.txt that says "
            "'MCP standardises tool access', then read it back to confirm."
        )
        print_user_question(q)
        answer = agent.answer(q)
        print_final_answer(answer)
        _cost_note(agent.llm.llm_calls)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Demo 4 — Agent uses search_web over MCP
# ---------------------------------------------------------------------------


def demo_4() -> None:
    print_section("Demo 4 — Agent uses search_web over MCP (offline corpus)")
    client = _new_client()
    try:
        agent = MCPAgent(client, llm=LLMClient(), on_event=handle_event)
        q = "Search the web for what MCP transports exist and summarise."
        print_user_question(q)
        answer = agent.answer(q)
        print_final_answer(answer)
        _cost_note(agent.llm.llm_calls)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Demo 5 — Same server, a SECOND independent client (write-once, plug-in-anywhere)
# ---------------------------------------------------------------------------


def demo_5() -> None:
    print_section("Demo 5 — One server, many clients + the protocol landscape")
    print("  A totally separate 6-line client connects to the SAME server.")
    client = _new_client()
    try:
        specs = client.list_tools()
        print(f"  Second client discovered the same {len(specs)} tools.")
        out = client.call_tool("search_web", {"query": "what is MCP", "max_results": 1})
        print_final_answer(out)
    finally:
        client.close()

    print("\n  The agent in Demos 2-4 and this raw client share ONE server.")
    print("  To expose the same tools to Claude Desktop, you add this config —")
    print('    "fellowship-tools": { "command": "python", "args": ["server.py"] }')
    print("  — and Claude Desktop becomes a third client. No tool code changes.")

    print_landscape()


_DEMOS = {"1": demo_1, "2": demo_2, "3": demo_3, "4": demo_4, "5": demo_5}


def _main(argv) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    fn = _DEMOS.get(argv[0].strip())
    if fn is None:
        print(f"Unknown demo '{argv[0]}'. Try one of: {', '.join(_DEMOS)}")
        return 2
    fn()
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
