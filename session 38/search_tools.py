"""
Session 38 — search_tools.py

REAL online web search. No canned corpus, no fake mode.

Two providers, tried in order:
  1. Tavily — a search API built for LLMs. Returns clean snippets plus a
     per-result relevance score. Needs TAVILY_API_KEY (free tier available).
  2. DuckDuckGo via the `ddgs` package — no API key at all. The zero-setup
     fallback so a student without a Tavily key can still run the workflow.

Whatever the provider, web_search() returns the SAME normalised shape so the
rest of the graph never has to care which engine answered:

    [{"title": str, "url": str, "content": str, "score": float}, ...]
"""

from __future__ import annotations

import os


def web_search(query: str, max_results: int = 4) -> list:
    """Run a real web search and return normalised hits.

    Picks Tavily if a key is present, else DuckDuckGo. Both paths are real
    network calls — this is why the workflow must be run with internet access.
    """
    if os.environ.get("TAVILY_API_KEY"):
        try:
            return _search_tavily(query, max_results)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the graph
            print(f"[search] Tavily failed ({exc}); falling back to DuckDuckGo.")
    return _search_duckduckgo(query, max_results)


def _search_tavily(query: str, max_results: int) -> list:
    """Tavily path. Carries a genuine 0-1 relevance score per result."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query=query, max_results=max_results)
    hits = []
    for item in response.get("results", []):
        hits.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": float(item.get("score", 0.0)),
        })
    return hits


def _search_duckduckgo(query: str, max_results: int) -> list:
    """No-key path via the `ddgs` package. DuckDuckGo gives no relevance
    score, so we leave it at 0.0 — the LLM quality gate is what judges the
    results anyway, not the engine's own ranking."""
    from ddgs import DDGS

    hits = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            hits.append({
                "title": item.get("title", ""),
                "url": item.get("href", item.get("url", "")),
                "content": item.get("body", item.get("content", "")),
                "score": 0.0,
            })
    return hits
