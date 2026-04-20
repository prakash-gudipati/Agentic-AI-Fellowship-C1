"""
Session 10 — Python Capstone: CLI Productivity Tool
Module: quote_api.py — Motivational quote fetcher via HTTP

DEMONSTRATES:
  S8 (APIs + HTTP requests) — GET request with the requests library
  S8 (Environment variables) — optional API key loaded from .env via os.environ
  S3 (Error handling)        — specific exception types, graceful fallback
  S3 (Single-responsibility) — this module does one thing: fetch a quote

PROD PATTERN: Every external API call must have:
  1. A timeout — so a slow server never hangs the program
  2. Specific exception handling — so the right error is caught
  3. A fallback — so the app works even when the network is unavailable
"""

import os
import requests

from config import QUOTE_API_URL


# Built-in fallback — the app stays useful even without internet
FALLBACK_QUOTE = {
    "q": "The secret of getting ahead is getting started.",
    "a": "Mark Twain",
}


def get_motivational_quote() -> dict:
    """
    Fetch one random motivational quote from the ZenQuotes API.

    ZenQuotes (zenquotes.io) is free for low-volume use without a key.
    If a ZENQUOTES_API_KEY is set in your .env, it will be sent as a header
    to access higher rate limits.

    Returns:
        dict with keys:
          "q" — the quote text
          "a" — the author's name
    """
    api_key = os.environ.get("ZENQUOTES_API_KEY", "")
    headers = {"apikey": api_key} if api_key else {}

    # PROD PATTERN (S3): crash before fix — show specific exception types
    try:
        response = requests.get(QUOTE_API_URL, headers=headers, timeout=5)
        response.raise_for_status()   # raises HTTPError for 4xx / 5xx responses

        data = response.json()
        if data and isinstance(data, list):
            return data[0]            # API returns a list; take the first item
        return FALLBACK_QUOTE

    except requests.exceptions.ConnectionError:
        print("  [INFO] No internet connection — using built-in quote.")
        return FALLBACK_QUOTE

    except requests.exceptions.Timeout:
        print("  [INFO] Quote API timed out — using built-in quote.")
        return FALLBACK_QUOTE

    except requests.exceptions.HTTPError as error:
        print(f"  [INFO] Quote API error ({error}) — using built-in quote.")
        return FALLBACK_QUOTE

    except requests.exceptions.RequestException as error:
        print(f"  [INFO] Could not fetch quote ({error}) — using built-in quote.")
        return FALLBACK_QUOTE
