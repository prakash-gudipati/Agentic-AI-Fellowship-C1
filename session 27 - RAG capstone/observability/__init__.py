"""Session 26 — observability package.

THE PATTERN — the score told you WHAT broke. The trace tells you WHY.
The cost tells you HOW MUCH the fix is allowed to cost.

Modules:

  langsmith_tracing.py  LangSmith wrapper + offline JSONL fallback.
                         Exposes the `track` decorator (turns any
                         function into a traced LangSmith run) and
                         `wrap_llm_client` (auto-instruments the
                         Anthropic / OpenAI SDK so token counts + cost
                         flow through automatically).

  cost_budget.py        Daily-budget enforcement. Pulls the running
                         24-hour spend from the LangSmith API and fails
                         CI if the projection exceeds the threshold.
                         Falls back to the offline JSONL log when
                         LangSmith isn't reachable.

  debugger.py           Trace replay. Given a failed run_id, fetch the
                         original inputs from LangSmith, re-run locally
                         with an alternative model / retriever, print a
                         side-by-side diff (latency / cost / answer).

DROPPED in the v2 (LangSmith) build:
  - opik_tracing.py     -- LangSmith replaces it.
  - cost_tracker.py     -- LangSmith captures tokens + cost from the
                          provider response. No CostMeter needed.
  - dashboard.py        -- LangSmith's hosted UI replaces our custom
                          terminal dashboard.

Why the deletes? S26's production lesson is "use the platform, not
custom code." Anything LangSmith does for free, we don't rebuild.

The LangSmith wrapper falls back to a JSONL file when the `langsmith`
package isn't installed or no `LANGCHAIN_API_KEY` is set. The fallback
captures the same shape LangSmith would persist -- root + nested runs,
timing, inputs/outputs, token + cost estimates. This keeps the demo
runnable on a fresh laptop without a LangSmith account.
"""

from .langsmith_tracing import (
    track,
    wrap_llm_client,
    trace_span,
    set_current_run_metrics,
    is_langsmith_active,
    get_offline_tracer,
    get_project_url,
    get_run_url,
)
from .cost_budget import (
    check_budget,
    check_budget_synthetic,
    BudgetExceededError,
    BudgetReport,
)
from .debugger import replay, fetch_original_run, ReplayResult

__all__ = [
    # tracing
    "track",
    "wrap_llm_client",
    "trace_span",
    "set_current_run_metrics",
    "is_langsmith_active",
    "get_offline_tracer",
    "get_project_url",
    "get_run_url",
    # cost budget gate
    "check_budget",
    "check_budget_synthetic",
    "BudgetExceededError",
    "BudgetReport",
    # debugger
    "replay",
    "fetch_original_run",
    "ReplayResult",
]
