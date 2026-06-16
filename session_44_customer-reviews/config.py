"""config.py — single source of all settings for Review Radar.

Every other module imports its settings from here (Convention 1). Secrets are read
from the environment, never hardcoded (Convention 2). A local `.env` is loaded for
dev convenience; in production the host injects real env vars and load_dotenv() is a
harmless no-op.
"""

import os

from dotenv import load_dotenv

# Populate os.environ from a local .env if one exists. Safe no-op when absent
# (e.g. on the serverless host, where real env vars are already set).
load_dotenv()


def _as_bool(value: str | None) -> bool:
    """Interpret a boolean-ish env string. Accept the usual truthy spellings so
    FAKE_LLM=1 / true / yes all enable offline mode."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


# --- Secrets / model (read from environment; never hardcoded) ---

# Required only when FAKE_LLM is off (i.e. real Claude calls). Empty otherwise.
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# Which Claude model to call. Defaults to Claude Haiku per CLAUDE.md.
MODEL: str = os.environ.get("MODEL", "claude-haiku-4-5-20251001")

# Offline mode: return canned analysis so the app runs with no API key (Convention 5).
FAKE_LLM: bool = _as_bool(os.environ.get("FAKE_LLM"))


# --- Tunable constants (business rules from the spec, kept in one place) ---

# Themes returned per batch: aim for MIN..MAX, never exceed MAX (AC4).
MIN_THEMES: int = 3
MAX_THEMES: int = 5

# Best-effort cap: above this, analyze_reviews refuses explicitly rather than
# silently truncating (AC10).
LARGE_BATCH_LIMIT: int = 300
