"""
Session 26 -- observability/cost_budget.py

THE PATTERN -- turn $/day spend into a CI gate, using LangSmith as the
source of truth for actual cost.

WHY THIS FILE
-------------
The S25 eval gate said "fail the build if faithfulness drops." This
file says "fail the build if the PR projects to cost more than $X/day."

Same shape. Same code-review story. Different number.

WHERE THE COST DATA COMES FROM
------------------------------
LangSmith automatically captures token counts and dollar cost from
every wrapped LLM call (because we used `wrap_anthropic(...)` /
`@traceable`). This file just READS those numbers -- via the LangSmith
Client API -- and gates the build on them.

When LangSmith isn't reachable, the same gate falls back to the offline
JSONL trace file produced by `langsmith_tracing.get_offline_tracer()`.

HOW IT WORKS
------------
1. Pull the last N runs in the current LangSmith project.
2. Sum cost across those runs.
3. Project to daily spend using an explicit `queries_per_day` assumption.
4. If projected daily spend > threshold, raise BudgetExceededError.

PRODUCTION PATTERNS INTRODUCED (S26):
  - cost-as-CI-gate                             (NEW, named today)
  - pull-from-platform vs custom-meter          (NEW, named today)
  - extrapolation from a dev sample to a
    daily projection                            (NEW, named today)
  - thresholding with pass / warn / fail tiers  (NEW, named today)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .langsmith_tracing import (
    is_langsmith_active,
    get_offline_tracer,
    DEFAULT_PROJECT,
)


class BudgetExceededError(RuntimeError):
    """Raised when a budget gate fails. CI returns non-zero."""


# Default thresholds (override via env or args in production).
DEFAULT_DAILY_BUDGET_USD: float = float(
    os.environ.get("RAG_DAILY_BUDGET_USD", "5.00"))
WARN_FRACTION: float = 0.75    # warn at 75% of budget
FAIL_FRACTION: float = 1.00    # fail at 100% of budget


@dataclass
class BudgetReport:
    """The result of a budget check."""
    source: str               # "langsmith" | "offline" | "synthetic"
    daily_budget_usd: float
    sample_cost_usd: float
    sample_queries: int
    queries_per_day: int
    projected_daily_usd: float
    headroom_usd: float
    tier: str                 # "pass" | "warn" | "fail"
    message: str

    def is_pass(self) -> bool:
        return self.tier == "pass"

    def is_warn(self) -> bool:
        return self.tier == "warn"

    def is_fail(self) -> bool:
        return self.tier == "fail"


# Pull sample cost -- from LangSmith if active, else offline JSONL.

def _pull_sample_from_langsmith(project: str,
                                last_n: int) -> tuple:
    """Pull the last N runs in `project` and sum total_cost."""
    from langsmith import Client                       # type: ignore
    client = Client()
    runs = list(client.list_runs(
        project_name=project, execution_order=1, limit=last_n))
    sample_cost = 0.0
    sample_n = 0
    for r in runs:
        cost = getattr(r, "total_cost", None)
        if cost is None:
            cost = (getattr(r, "extra", {}) or {}).get("cost_usd", 0.0)
        try:
            sample_cost += float(cost or 0.0)
            sample_n += 1
        except (TypeError, ValueError):
            continue
    return sample_cost, sample_n


def _pull_sample_offline(last_n: int) -> tuple:
    """Pull the last N root runs from the offline JSONL fallback."""
    runs = get_offline_tracer().get_recent(n=last_n, only_roots=True)
    sample_cost = 0.0
    sample_n = 0
    for r in runs:
        try:
            sample_cost += float(r.get("cost_usd") or 0.0)
            sample_n += 1
        except (TypeError, ValueError):
            continue
    return sample_cost, sample_n


def _classify(fraction: float) -> str:
    if fraction >= FAIL_FRACTION:
        return "fail"
    if fraction >= WARN_FRACTION:
        return "warn"
    return "pass"


def _print_report(report: "BudgetReport") -> None:
    msg_lines = report.message.split("\n  ")
    print()
    print("=" * 64)
    print("  COST BUDGET GATE  -  " + report.tier.upper())
    print("=" * 64)
    for line in msg_lines:
        print("  " + line)
    print("=" * 64)


def check_budget(*,
                 last_n: int = 10,
                 queries_per_day: int = 10_000,
                 daily_budget_usd: float = DEFAULT_DAILY_BUDGET_USD,
                 project: str = DEFAULT_PROJECT,
                 raise_on_fail: bool = True) -> BudgetReport:
    """Project daily spend from the last `last_n` traced runs and gate
    against the daily budget. Reads from LangSmith if active, else from
    the offline JSONL log.
    """
    if last_n <= 0:
        raise ValueError("last_n must be > 0 (got " + str(last_n) + ")")
    if daily_budget_usd <= 0:
        raise ValueError(
            "daily_budget_usd must be > 0 (got "
            + str(daily_budget_usd) + ")")

    if is_langsmith_active():
        source = "langsmith"
        try:
            sample_cost, sample_n = _pull_sample_from_langsmith(
                project=project, last_n=last_n)
        except Exception as e:
            print("[budget] WARN: LangSmith pull failed ("
                  + str(e) + ") -- falling back to offline log.")
            source = "offline"
            sample_cost, sample_n = _pull_sample_offline(last_n)
    else:
        source = "offline"
        sample_cost, sample_n = _pull_sample_offline(last_n)

    if sample_n == 0:
        raise RuntimeError(
            "No runs found in " + source + ". Run the pipeline at "
            "least once before invoking the budget gate.")

    cost_per_query = sample_cost / float(sample_n)
    projected_daily = cost_per_query * float(queries_per_day)
    headroom = daily_budget_usd - projected_daily
    fraction = projected_daily / daily_budget_usd
    tier = _classify(fraction)

    msg = (
        "source:  " + source + "\n  "
        + "sample:  " + str(sample_n) + " runs  cost $"
        + ("%.4f" % sample_cost) + "  ($"
        + ("%.5f" % cost_per_query) + "/query)\n  "
        + "traffic: " + ("{:,}".format(queries_per_day))
        + " queries/day\n  "
        + "projected daily spend: $" + ("%.2f" % projected_daily)
        + "  (budget $" + ("%.2f" % daily_budget_usd)
        + ", headroom $" + ("%+.2f" % headroom) + ")\n  "
        + "tier:    " + tier.upper()
        + "  (" + ("%.0f" % (fraction * 100.0)) + "% of budget)"
    )

    report = BudgetReport(
        source=source,
        daily_budget_usd=daily_budget_usd,
        sample_cost_usd=sample_cost,
        sample_queries=sample_n,
        queries_per_day=queries_per_day,
        projected_daily_usd=projected_daily,
        headroom_usd=headroom,
        tier=tier,
        message=msg,
    )
    _print_report(report)

    if tier == "fail" and raise_on_fail:
        raise BudgetExceededError(
            "projected daily spend $" + ("%.2f" % projected_daily)
            + " > budget $" + ("%.2f" % daily_budget_usd))

    return report


def check_budget_synthetic(sample_cost_usd: float,
                           sample_queries: int,
                           *,
                           queries_per_day: int = 10_000,
                           daily_budget_usd: float = DEFAULT_DAILY_BUDGET_USD,
                           raise_on_fail: bool = True) -> BudgetReport:
    """Run the budget math on hand-supplied numbers -- no LangSmith call.
    Useful in the live demo to force the FAIL tier on a deliberately-
    expensive scenario without burning real tokens.
    """
    if sample_queries <= 0:
        raise ValueError(
            "sample_queries must be > 0 (got "
            + str(sample_queries) + ")")
    cost_per_query = sample_cost_usd / float(sample_queries)
    projected_daily = cost_per_query * float(queries_per_day)
    headroom = daily_budget_usd - projected_daily
    fraction = projected_daily / daily_budget_usd
    tier = _classify(fraction)

    msg = (
        "source:  synthetic (no LLM calls made)\n  "
        + "sample:  " + str(sample_queries) + " queries  cost $"
        + ("%.4f" % sample_cost_usd) + "  ($"
        + ("%.5f" % cost_per_query) + "/query)\n  "
        + "traffic: " + ("{:,}".format(queries_per_day))
        + " queries/day\n  "
        + "projected daily spend: $" + ("%.2f" % projected_daily)
        + "  (budget $" + ("%.2f" % daily_budget_usd)
        + ", headroom $" + ("%+.2f" % headroom) + ")\n  "
        + "tier:    " + tier.upper()
        + "  (" + ("%.0f" % (fraction * 100.0)) + "% of budget)"
    )

    report = BudgetReport(
        source="synthetic",
        daily_budget_usd=daily_budget_usd,
        sample_cost_usd=sample_cost_usd,
        sample_queries=sample_queries,
        queries_per_day=queries_per_day,
        projected_daily_usd=projected_daily,
        headroom_usd=headroom,
        tier=tier,
        message=msg,
    )
    _print_report(report)

    if tier == "fail" and raise_on_fail:
        raise BudgetExceededError(
            "projected daily spend $" + ("%.2f" % projected_daily)
            + " > budget $" + ("%.2f" % daily_budget_usd))

    return report


if __name__ == "__main__":
    print("\n--- Scenario 1: a happy build ---")
    check_budget_synthetic(sample_cost_usd=0.012, sample_queries=10,
                           queries_per_day=10_000,
                           daily_budget_usd=15.00, raise_on_fail=False)

    print("\n--- Scenario 2: warn (~80% of budget) ---")
    check_budget_synthetic(sample_cost_usd=0.012, sample_queries=10,
                           queries_per_day=10_000,
                           daily_budget_usd=1.50, raise_on_fail=False)

    print("\n--- Scenario 3: fail ---")
    try:
        check_budget_synthetic(sample_cost_usd=0.05, sample_queries=10,
                               queries_per_day=10_000,
                               daily_budget_usd=15.00, raise_on_fail=True)
    except BudgetExceededError as e:
        print("\n[demo] caught BudgetExceededError: " + str(e))
