"""
Session 30 - stateful_tools.py

The Tool dataclass in tools.py models tools as pure functions: same input,
same output, no side effects, no memory between calls. That covers most
tools - calculator, web_search, wikipedia_summary.

But some tools have STATE: a database cursor, an in-progress shopping cart,
a conversation handle to a third-party API. Calling them once changes what
calling them again means.

This module shows the StatefulTool pattern.

The trick: separate the IMMUTABLE schema (what the model reads) from the
MUTABLE session (what your code carries between calls). The model only ever
sees a session_id it doesn't have to understand; your code looks up the
real state from the session_id.

Why model the state OUTSIDE the LLM's view:
  - The LLM does not need to (and should not try to) hold session_id
    semantics. Just route it.
  - Easy to swap the backing store (in-memory dict -> Redis -> SQL) without
    touching the schema.
  - The session_id appears in action_log.jsonl, so the SRE can correlate
    every call against the same session.

Example shown below: a tiny shopping-cart tool that holds line items
across multiple calls within one agent run.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

from tools import Tool


# ── Session store ──────────────────────────────────────────────────────────
# In production this would be Redis / a database. For the session we keep
# state in an in-process dict, keyed by session_id.
@dataclass
class CartSession:
    cart_id: str
    items: List[Dict[str, Any]] = field(default_factory=list)


_CART_SESSIONS: Dict[str, CartSession] = {}


def new_cart_session() -> str:
    cart_id = f"cart-{uuid.uuid4().hex[:8]}"
    _CART_SESSIONS[cart_id] = CartSession(cart_id=cart_id)
    return cart_id


# ── Tool 1: cart_add_item ─────────────────────────────────────────────────
def _run_cart_add_item(args: Dict[str, Any]) -> str:
    cart_id = str(args.get("cart_id", "")).strip()
    sku = str(args.get("sku", "")).strip()
    qty = int(args.get("qty", 1))

    if cart_id not in _CART_SESSIONS:
        return (
            f"ERROR[invalid_input]: unknown cart_id '{cart_id}'. "
            "Call cart_new first to start a session, then pass that cart_id."
        )
    if not sku:
        return "ERROR[invalid_input]: sku is required."

    _CART_SESSIONS[cart_id].items.append({"sku": sku, "qty": qty})
    n = len(_CART_SESSIONS[cart_id].items)
    return f"Added {qty} x {sku} to cart {cart_id}. Cart now has {n} line items."


cart_add_item_tool = Tool(
    name="cart_add_item",
    description=(
        "Add a line item to an existing shopping cart session. "
        "Use this AFTER calling cart_new to get a cart_id. "
        "Pass the cart_id and the sku to add. "
        "Example: cart_id='cart-abc123', sku='BOOK-7732', qty=2."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "cart_id": {
                "type": "string",
                "description": "The session ID returned by cart_new. Always pass exactly the string cart_new gave you.",
            },
            "sku": {
                "type": "string",
                "description": "The SKU code of the product to add. Example: 'BOOK-7732'.",
            },
            "qty": {
                "type": "integer",
                "description": "How many of this SKU to add. Default 1.",
                "default": 1,
            },
        },
        "required": ["cart_id", "sku"],
    },
    run=_run_cart_add_item,
)


# ── Tool 2: cart_new (opens a session) ────────────────────────────────────
def _run_cart_new(args: Dict[str, Any]) -> str:
    cart_id = new_cart_session()
    return f"OK: opened cart session. cart_id={cart_id}. Pass this cart_id to cart_add_item and cart_summary."


cart_new_tool = Tool(
    name="cart_new",
    description=(
        "Open a new shopping cart session. Returns a cart_id you must pass to "
        "every subsequent cart_add_item and cart_summary call. "
        "Use this ONCE at the start of a multi-step cart-building task."
    ),
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
    run=_run_cart_new,
)


# ── Tool 3: cart_summary (reads session state) ────────────────────────────
def _run_cart_summary(args: Dict[str, Any]) -> str:
    cart_id = str(args.get("cart_id", "")).strip()
    if cart_id not in _CART_SESSIONS:
        return f"ERROR[invalid_input]: unknown cart_id '{cart_id}'."
    session = _CART_SESSIONS[cart_id]
    if not session.items:
        return f"Cart {cart_id} is empty."
    lines = [f"Cart {cart_id} contents:"]
    for item in session.items:
        lines.append(f"  - {item['qty']} x {item['sku']}")
    return "\n".join(lines)


cart_summary_tool = Tool(
    name="cart_summary",
    description=(
        "Read the current contents of a shopping cart session. "
        "Use this to confirm what's been added so far. "
        "Pass the cart_id returned by cart_new."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "cart_id": {
                "type": "string",
                "description": "The session ID returned by cart_new.",
            },
        },
        "required": ["cart_id"],
    },
    run=_run_cart_summary,
)


STATEFUL_CART_TOOLS = [cart_new_tool, cart_add_item_tool, cart_summary_tool]


# ── KEY PATTERN to notice ─────────────────────────────────────────────────
# The descriptions reference each other - cart_new says "Returns a cart_id
# you must pass to every subsequent cart_add_item and cart_summary call."
# That's the CONTRAST pattern at work for stateful tools: each description
# tells the model where this tool fits in the sequence.
#
# Without that cross-reference, the model has no way to know cart_add_item
# requires a prior cart_new call. With it, the model reads all three
# descriptions, builds a tiny mental state machine, and calls them in order.


if __name__ == "__main__":
    print("=== Stateful tool demo ===")
    open_msg = _run_cart_new({})
    print(open_msg)
    cart_id = open_msg.split("cart_id=")[1].split(".")[0]

    print(_run_cart_add_item({"cart_id": cart_id, "sku": "BOOK-7732", "qty": 2}))
    print(_run_cart_add_item({"cart_id": cart_id, "sku": "PEN-001", "qty": 5}))
    print(_run_cart_summary({"cart_id": cart_id}))
