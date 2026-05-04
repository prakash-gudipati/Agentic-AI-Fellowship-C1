"""
llm_client — Session 16 production-grade LLM API wrapper.

S16 extends the S15 wrapper with FOUR new capabilities:
  1. Prompt caching (cache_system_prompts default True)
  2. Native structured outputs (complete_structured)
  3. Reasoning model routing (complete_reasoning)
  4. Indirect-injection defense (safe_complete)

Public API:

    from llm_client import LLMClient, Conversation
    from llm_client.errors import (
        LLMError, ProviderTimeout, ProviderUnavailable, RateLimitHit,
        InvalidAPIKey, ContextLengthExceeded, ContentFilterTriggered,
        BudgetExceeded, AllProvidersFailed,
        # S16 new:
        PromptInjectionDetected, StructuredOutputFailed, ReasoningTimeout,
    )
    from llm_client.security import (
        sanitize_input, spotlight, output_filter, safe_combine_documents,
    )
    from llm_client.structured import (
        to_openai_response_format, to_anthropic_tool, parse_response,
    )
"""

from .client import LLMClient
from .conversation import Conversation
from .cost import calculate_cost_usd, USD_PER_MTOK, CACHED_INPUT_RATE_MULTIPLIER
from . import errors, security, structured

__all__ = [
    "LLMClient",
    "Conversation",
    "calculate_cost_usd",
    "USD_PER_MTOK",
    "CACHED_INPUT_RATE_MULTIPLIER",
    "errors",
    "security",
    "structured",
]
__version__ = "1.1.0"  # bumped from S15's 1.0.0
