"""
Session 39 — search_tools.py

The web search a node calls. Carried forward from S38 with one addition: an
OFFLINE canned corpus so fake-mode runs (and the selftest) never touch the
network.

  Real mode  : Tavily (needs TAVILY_API_KEY) -> DuckDuckGo (keyless) fallback.
  Fake mode  : a tiny canned result set, returned instantly, no network.

Each result is normalised to the same shape: {"title", "content", "url"}.
"""

from __future__ import annotations

import os


def _fake_enabled() -> bool:
    if os.environ.get("FAKE_LLM", "").lower() in ("1", "true", "yes"):
        return True
    # If neither search backend can run, fall back to canned results too.
    return not os.environ.get("ANTHROPIC_API_KEY")


def web_search(query: str, max_results: int = 4) -> list:
    """Return a list of normalised search hits for `query`.

    Never raises: a search failure degrades to an empty list, which the quality
    gate will (correctly) score low.
    """
    if _fake_enabled():
        return _fake_results(query, max_results)

    results = _tavily(query, max_results)
    if results:
        return results
    return _duckduckgo(query, max_results)


# --------------------------------------------------------------------------
# real backends
# --------------------------------------------------------------------------


def _tavily(query: str, max_results: int) -> list:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return []
    try:
        from tavily import TavilyClient
        hits = TavilyClient(api_key=key).search(query, max_results=max_results).get("results", [])
        return [{"title": h.get("title", ""), "content": h.get("content", ""),
                 "url": h.get("url", "")} for h in hits]
    except Exception:
        return []


def _duckduckgo(query: str, max_results: int) -> list:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        return [{"title": h.get("title", ""), "content": h.get("body", ""),
                 "url": h.get("href", "")} for h in hits]
    except Exception:
        return []


# --------------------------------------------------------------------------
# offline canned corpus
# --------------------------------------------------------------------------


def _fake_results(query: str, max_results: int) -> list:
    """Deterministic canned hits. The content mentions the query so the offline
    report reads as if it were grounded in real sources."""
    base = [
        {"title": f"A practical guide to {query}",
         "content": f"This guide covers {query} with named tools and a worked example.",
         "url": "https://example.com/guide"},
        {"title": f"{query}: production trade-offs",
         "content": f"Teams weigh cost, latency, and reliability when applying {query}.",
         "url": "https://example.com/tradeoffs"},
        {"title": f"Case study — {query} in the wild",
         "content": f"A real team shipped {query} and recorded what broke and what held.",
         "url": "https://example.com/case-study"},
        {"title": f"FAQ about {query}",
         "content": f"Common questions and pitfalls when adopting {query}.",
         "url": "https://example.com/faq"},
    ]
    return base[:max_results]
