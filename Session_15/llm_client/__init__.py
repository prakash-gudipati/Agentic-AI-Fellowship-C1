"""
llm_client — production-grade LLM API wrapper for the Agentic AI Fellowship.

Public API:

    from llm_client import LLMClient, Conversation
    from llm_client.errors import (
        LLMError, ProviderTimeout, ProviderUnavailable, RateLimitHit,
        InvalidAPIKey, ContextLengthExceeded, ContentFilterTriggered,
        BudgetExceeded, AllProvidersFailed,
    )

Quickstart:

    client = LLMClient(provider="anthropic")
    print(client.complete("What is a vending machine?"))
    print("Spent so far: $", client.total_cost_usd)
"""

from .client import LLMClient
from .conversation import Conversation
from .cost import calculate_cost_usd, USD_PER_MTOK
from . import errors

__all__ = [
    "LLMClient",
    "Conversation",
    "calculate_cost_usd",
    "USD_PER_MTOK",
    "errors",
]
__version__ = "1.0.0"
