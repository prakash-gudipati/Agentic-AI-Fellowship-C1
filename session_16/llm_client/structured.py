"""
Structured outputs — Session 16 NEW.

Replaces the S14 "ask for JSON in the prompt + parse + retry on failure"
pattern with the providers' native structured-output APIs:

  - OpenAI: response_format={"type": "json_schema", "schema": ..., "strict": True}
            Schema-guaranteed at the API level — invalid output is
            literally impossible.
  - Anthropic: tool_use mode with a single forced tool call. The model's
              "tool input" IS the structured response. Same guarantee.

The wrapper exposes ONE method — complete_structured(prompt, schema=PydanticModel)
— that picks the right provider mechanism and returns a validated
Pydantic instance.

Why Pydantic? Because:
  1. The schema doubles as documentation.
  2. Type checkers (mypy) understand it.
  3. Pydantic's .model_json_schema() generates the exact JSON Schema
     shape both providers want.
"""

from __future__ import annotations

import json
from typing import Any, Type, TypeVar

try:
    from pydantic import BaseModel, ValidationError
except ImportError as e:
    raise ImportError(
        "S16 structured outputs require pydantic>=2.0. Install with: "
        "pip install 'pydantic>=2.0'"
    ) from e

from .errors import StructuredOutputFailed


T = TypeVar("T", bound=BaseModel)


def to_openai_response_format(schema: Type[BaseModel]) -> dict:
    """Convert a Pydantic model into OpenAI's response_format dict.

    OpenAI's strict mode requires every property to be required and
    additionalProperties: false. Pydantic's default schema mostly works,
    but we tighten it to be safe.
    """
    json_schema = schema.model_json_schema()
    # OpenAI strict mode: every field required, no extra properties
    json_schema = _make_strict(json_schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": json_schema,
            "strict": True,
        },
    }


def to_anthropic_tool(schema: Type[BaseModel], tool_name: str = "respond") -> dict:
    """Convert a Pydantic model into an Anthropic tool definition.

    To force structured output on Anthropic we define ONE tool whose
    input_schema matches our Pydantic model. We then force the model to
    call this tool via tool_choice. The model's tool input IS the
    response.
    """
    return {
        "name": tool_name,
        "description": f"Return a {schema.__name__} object.",
        "input_schema": schema.model_json_schema(),
    }


def parse_response(text: str, schema: Type[T]) -> T:
    """Parse a JSON-string response into a validated Pydantic instance.

    Raises StructuredOutputFailed if the JSON does not validate.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise StructuredOutputFailed(
            f"Response was not valid JSON: {e}"
        ) from e
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        raise StructuredOutputFailed(
            f"Response did not match schema {schema.__name__}: {e}"
        ) from e


def parse_tool_input(tool_input: Any, schema: Type[T]) -> T:
    """Parse Anthropic tool_use.input (already a dict) into Pydantic."""
    try:
        return schema.model_validate(tool_input)
    except ValidationError as e:
        raise StructuredOutputFailed(
            f"Tool input did not match schema {schema.__name__}: {e}"
        ) from e


def _make_strict(node: dict) -> dict:
    """Recursively tighten a JSON Schema for OpenAI strict mode.

    - Every object becomes additionalProperties: false
    - Every property is added to required
    - Recurses into nested objects and array items
    """
    if not isinstance(node, dict):
        return node
    if node.get("type") == "object" and "properties" in node:
        node = dict(node)
        node["additionalProperties"] = False
        node["required"] = list(node["properties"].keys())
        node["properties"] = {k: _make_strict(v) for k, v in node["properties"].items()}
    if node.get("type") == "array" and "items" in node:
        node = dict(node)
        node["items"] = _make_strict(node["items"])
    if "$defs" in node:
        node = dict(node)
        node["$defs"] = {k: _make_strict(v) for k, v in node["$defs"].items()}
    return node
