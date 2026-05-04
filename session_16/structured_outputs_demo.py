"""
Demo 2 — Native structured outputs.

Defines a Pydantic schema for a Customer record. Asks the model to fill
it from a free-form description. Gets back a validated Pydantic instance.

The S14 way: ask for JSON in the prompt, parse, retry on parse failure.
The S16 way: native structured outputs. Schema is enforced at the API
level. Invalid output is literally not possible.

Run:
    $ python -m examples.structured_outputs_demo
"""

from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from llm_client import LLMClient


class Customer(BaseModel):
    """A normalised customer record extracted from free-form text."""
    full_name: str = Field(..., description="Customer's full legal name")
    email: str = Field(..., description="Email address, lowercased")
    plan: Literal["free", "starter", "pro", "enterprise"]
    seats: int = Field(..., ge=1, description="Number of paid seats; min 1")
    is_active: bool


client = LLMClient(provider="anthropic")

DESCRIPTION = (
    "Just signed up: Aisha Khan (aisha.khan@example.com). She picked the "
    "Pro plan and is paying for 12 users. Onboarding starts Monday."
)

result = client.complete_structured(
    f"Extract a Customer record from this text: {DESCRIPTION}",
    schema=Customer,
)

print(f"Type:        {type(result).__name__}")
print(f"Full name:   {result.full_name}")
print(f"Email:       {result.email}")
print(f"Plan:        {result.plan}")
print(f"Seats:       {result.seats}")
print(f"Is active:   {result.is_active}")
print(f"\nValidated:   {result.model_dump_json(indent=2)}")
print(f"Cost so far: ${client.total_cost_usd:.6f}")
