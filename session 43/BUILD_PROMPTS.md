# Session 43 — The Live Build Prompt Sequence
# (Type these into Claude Code inside the Claude Desktop app, in order.)

## STEP 0 — Set up (no prompt yet)
1. Open the Claude Desktop app. Open Claude Code on an empty folder named `smart-summarizer`.
2. WRITE CLAUDE.md FIRST — before any prompt. (Paste the CLAUDE.md from the slide,
   or type `/init` and let Claude draft one, then edit it to match.)

## PROMPT 1 — Scaffold
Read CLAUDE.md. Scaffold the project: create config.py that reads ANTHROPIC_API_KEY
and MODEL from the environment, a main.py with a FastAPI app and a GET /health endpoint
that returns {"status": "ok"}, and requirements.txt with fastapi, uvicorn, anthropic,
pydantic. Do NOT add the summarizer yet. Then run it locally and show me /health works.

## PROMPT 2 — The summarizer endpoint
Add the summarizer. Create summarizer.py with summarize(text: str) -> dict that calls
Claude Haiku via the Anthropic SDK and returns {"tl_dr": str, "key_points": [str, str, str]}
with exactly 3 points. Follow CLAUDE.md: read settings from config.py, ask the model for
JSON, and never crash if parsing fails. Add a POST /summarize endpoint to main.py that takes
{"text": str}. Add a FAKE_LLM mode so I can test without an API key.

## PROMPT 3 — The web form
Add a GET / endpoint to main.py that returns a plain HTML page: a textarea, a Summarize
button, and JavaScript that POSTs the text to /summarize and shows the TL;DR and the 3 key
points. No framework — inline HTML and CSS only.

## PROMPT 4 — Vercel deploy config
Prepare this for Vercel. Add vercel.json using the @vercel/python runtime, an api/index.py
that imports the FastAPI app from main.py, a .gitignore that ignores .env and __pycache__,
and a .env.example. Do not put my real key anywhere.

## STEP 5 — Ship (no prompt — you do this)
1. git init, commit, push to a new GitHub repo.
2. vercel.com -> Add New Project -> import the repo.
3. Settings -> Environment Variables -> add ANTHROPIC_API_KEY (your real key).
4. Deploy. Open the live URL. Paste text. Watch it summarize.
