# Smart Summarizer API — Session 43 reference build

A FastAPI web app that turns pasted text into a TL;DR + 3 key points using Claude Haiku,
deployable to a live Vercel URL. Built live by directing Claude Code in the Claude Desktop app.

## Files
- `CLAUDE.md`     — the project briefing you write FIRST (Claude Code reads it before coding).
- `config.py`     — Config File Separation: every setting, read from the environment.
- `summarizer.py` — `summarize(text) -> {tl_dr, key_points[3]}`; calls Claude Haiku.
- `main.py`       — FastAPI app: `GET /` (HTML form), `GET /health`, `POST /summarize`.
- `api/index.py`  — Vercel serverless entry point.
- `vercel.json`   — Vercel build/route config (@vercel/python).
- `BUILD_PROMPTS.md` — the exact prompt sequence used in the session.

## Run locally (offline, no API key)
```
pip install -r requirements.txt
FAKE_LLM=1 uvicorn main:app --reload
# open http://127.0.0.1:8000
```

## Run locally with the real model
```
cp .env.example .env      # put your real ANTHROPIC_API_KEY in .env
export $(cat .env | xargs)
uvicorn main:app --reload
```

## Deploy to Vercel
1. Push to GitHub.
2. Import the repo at vercel.com.
3. Add `ANTHROPIC_API_KEY` under Settings -> Environment Variables.
4. Deploy -> you get a live URL.

## Production patterns shown
- Config File Separation (from Session 42).
- Secrets via Environment Variables (this session): the key lives in `.env` locally and in
  Vercel's env-var panel in production. It is never written in code and never committed.
