# CLAUDE.md — Review Radar (Customer Review Analyzer)

This file is the project briefing. Claude Code reads it before writing any code.
The full intent lives in the spec files under `specs/` — this file is the short version.

## What we are building
Review Radar: a web app for a product team. A user pastes a batch of customer
reviews (one per line). The app returns, for the whole batch:
- overall sentiment split (how many positive / negative / neutral),
- the top recurring themes / complaints,
- a short plain-English summary a product manager can act on.
It ships as a live URL.

## Spec-Driven Development (how we work here)
We do NOT vibe-code straight to a solution. We follow:
  specs/spec.md  →  specs/plan.md  →  specs/tasks.md  →  implement
The spec is the source of truth. If the code and the spec disagree, the spec wins —
fix the spec first, then the code. Every implementation prompt names the task it satisfies.

## Tech stack
- Python 3.10+
- FastAPI (web framework)
- Anthropic Python SDK (the AI brain) — model: Claude Haiku
- Plain HTML + inline CSS/JS for the page (no frontend framework)
- Deploy target: a serverless Python host (Vercel / Railway / Render)

## Architecture (keep it this simple)
- `config.py`    — ALL settings. Reads ANTHROPIC_API_KEY and MODEL from the environment.
- `analyzer.py`  — one function `analyze_reviews(reviews) -> dict` that calls Claude.
- `main.py`      — the FastAPI app: GET `/` (HTML form), GET `/health`, POST `/analyze`.

## Conventions (non-negotiable)
1. CONFIG FILE SEPARATION — every file imports settings from `config.py`. Never hardcode twice.
2. SECRETS VIA ENVIRONMENT VARIABLES — read the key from `os.environ`. Never commit `.env`.
3. Type hints. Small functions. Comments explain WHY, not WHAT.
4. The model is asked for JSON; the code NEVER crashes if parsing fails (fail soft).
5. A `FAKE_LLM=1` mode returns canned output so the app runs with no API key.

## Out of scope (do NOT add)
- No database. No auth. No file upload. No frontend framework. Plain HTML only.
