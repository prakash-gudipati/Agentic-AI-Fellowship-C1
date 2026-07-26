"""
Session 36 — protocol_landscape.py

A tiny reference, printed in Demo 5, that places MCP next to the other agent
protocols students will meet after graduation. This is the "brief landscape
view" the curriculum asks for — not a deep dive, just enough to navigate.

Facts verified May 2026. All three are open protocols.
"""

from __future__ import annotations

PROTOCOLS = [
    {
        "name": "MCP — Model Context Protocol",
        "origin": "Anthropic (2024)",
        "connects": "an agent ↔ TOOLS and data sources",
        "one_liner": (
            "The dominant agent-to-TOOL standard. One server exposes tools; "
            "any client can use them. This whole session."
        ),
    },
    {
        "name": "A2A — Agent-to-Agent",
        "origin": "Google",
        "connects": "an agent ↔ OTHER AGENTS",
        "one_liner": (
            "Lets agents on different frameworks delegate to each other. "
            "Sits one layer ABOVE MCP — A2A coordinates, MCP fetches."
        ),
    },
    {
        "name": "AG-UI — Agent-User Interaction",
        "origin": "open community",
        "connects": "an agent ↔ a USER-FACING APP",
        "one_liner": (
            "Streams agent updates into interactive UIs so agents escape "
            "plain-text chat. The user-facing layer."
        ),
    },
]


def print_landscape() -> None:
    print("\nWhere MCP sits in the 2026 protocol stack:")
    for p in PROTOCOLS:
        print(f"\n  {p['name']}  ({p['origin']})")
        print(f"    connects: {p['connects']}")
        print(f"    {p['one_liner']}")
    print(
        "\n  Rule of thumb: MCP = tools, A2A = other agents, AG-UI = the user.\n"
        "  They solve adjacent problems at adjacent layers — not competitors.\n"
    )
