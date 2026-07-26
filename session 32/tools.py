"""
Session 32 — tools.py

Two minimal tools so the demo agent has something to do besides chat.

The session topic is MEMORY, not tool design, so these tools are kept
deliberately small. Students who want sharper tools re-read S30
"Function Calling + Tool Design".

  - calculator(expression)        : safe-ish arithmetic eval
  - note_keeper(action, ...)      : in-process scratchpad — used in
                                    demo 4 to plant facts that the agent
                                    must then recall from memory in a
                                    later turn

These tools deliberately do NOT touch the memory layers. Tool side
effects belong to the tool. Memory belongs to the agent. Keeping the
boundary clean is half the reason production agents stay debuggable.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable, Dict, List


# ----------------------------------------------------------------------------
# calculator
# ----------------------------------------------------------------------------


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](
            _eval_node(node.left), _eval_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Disallowed expression node: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Safe arithmetic evaluator. Returns the result as a short string."""

    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree.body)
    except Exception as exc:
        return f"ERROR: bad expression: {exc}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


# ----------------------------------------------------------------------------
# note_keeper — in-process scratchpad
# ----------------------------------------------------------------------------


_NOTES: Dict[str, str] = {}


def note_keeper(action: str, key: str = "", value: str = "") -> str:
    """
    Tiny in-process key/value scratchpad.

    Actions:
      set <key> <value>  — store
      get <key>          — fetch
      list               — return all known keys
      clear              — reset

    Used by demo 4 to plant facts the agent has to recall LATER from
    semantic memory rather than from this dictionary.
    """

    action = action.strip().lower()
    if action == "set":
        if not key:
            return "ERROR: 'set' requires a key"
        _NOTES[key] = value
        return f"stored {key} = {value}"
    if action == "get":
        return _NOTES.get(key, f"ERROR: no such key: {key}")
    if action == "list":
        if not _NOTES:
            return "(empty)"
        return ", ".join(sorted(_NOTES.keys()))
    if action == "clear":
        _NOTES.clear()
        return "cleared"
    return f"ERROR: unknown action {action!r}"


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------


REGISTRY: Dict[str, Callable[..., str]] = {
    "calculator": calculator,
    "note_keeper": note_keeper,
}


def tool_names() -> List[str]:
    return sorted(REGISTRY.keys())
