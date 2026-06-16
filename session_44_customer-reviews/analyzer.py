"""analyzer.py — the AI brain: analyze_reviews(reviews) -> dict.

T2: input cleaning + counting (AC3) and the contract-shaped return.
T3: empty-batch and large-batch fail-soft guards (AC8, AC10).

The real Claude call, FAKE_LLM payload, and JSON validation arrive in T4-T6.
`analyze_reviews` must never raise to its caller — every path returns a dict in
the plan §2 contract shape.
"""

import config


def _clean_reviews(raw: str | list[str]) -> list[str]:
    """Split raw pasted text into one review per non-empty line (AC3).

    Accepts either the raw textarea string or an already-split list. Each line is
    stripped; blank / whitespace-only lines are dropped and never counted.
    """
    lines = raw if isinstance(raw, list) else raw.split("\n")
    return [stripped for line in lines if (stripped := line.strip())]


def _empty_contract(message: str | None = None) -> dict:
    """A contract-shaped result with zeroed analysis. Used for the empty and
    too-large fail-soft paths (and as the base shape elsewhere)."""
    return {
        "ok": False,
        "analyzed_count": 0,
        "sentiment": {"positive": 0, "negative": 0, "neutral": 0},
        "themes": [],
        "summary": "",
        "message": message,
    }


def analyze_reviews(raw: str | list[str]) -> dict:
    """Analyze a batch of reviews and return the plan §2 contract dict.

    Never raises. T3 implements the pre-call fail-soft guards; the real analysis
    (FAKE_LLM payload / Claude call / validation) is filled in by T4-T6.
    """
    reviews = _clean_reviews(raw)
    count = len(reviews)

    # AC8: nothing to analyze — return early, no model call.
    if count == 0:
        return _empty_contract(
            "There were no reviews to analyze. Paste one review per line."
        )

    # AC10: too large to process in full — refuse explicitly, do not truncate.
    if count > config.LARGE_BATCH_LIMIT:
        return _empty_contract(
            f"That batch is too large to process in full ({count} reviews). "
            "Try a smaller batch."
        )

    # Placeholder until T4-T6: a contract-shaped stub so callers see the right keys.
    return {
        "ok": True,
        "analyzed_count": count,
        "sentiment": {"positive": 0, "negative": 0, "neutral": 0},
        "themes": [],
        "summary": "",
        "message": None,
    }
