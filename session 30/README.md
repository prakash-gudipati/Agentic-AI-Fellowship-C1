# Session 30 — Function Calling + Tool Design

**Portfolio Session · Phase 5 · 85 min · Walkthrough Format (no live build)**

This is the reference build for Session 30 of the Agentic AI Builders
Fellowship. It is the **first agent in the course that uses native function
calling** — the structured tool-use APIs offered by Anthropic and OpenAI —
instead of the regex-parsed `Action:` lines from Session 29.

The same agent loop runs on **both providers**. The point of the session is
that once you describe a tool with a sharp JSON schema, the rest is wiring.

## What's in here

```
Code/
├─ agent.py                  ← provider-agnostic function-calling loop
├─ tools.py                  ← 3 research tools with sharp schemas
├─ schema_quality.py         ← VAGUE vs SHARP — the headline visual of the session
├─ providers/
│   ├─ __init__.py
│   ├─ base.py               ← abstract Provider interface
│   ├─ anthropic_provider.py ← uses messages.create + tool_use blocks
│   └─ openai_provider.py    ← uses chat.completions + tool_calls
├─ action_log.py             ← PROD PATTERN — structured agent action logging
├─ prompts.py                ← system prompt (shorter than S29 — format is now native)
├─ retry.py                  ← exponential-backoff retry (carry-over from S29)
├─ trace_logger.py           ← coloured terminal trace (carry-over from S29)
├─ demo.py                   ← four runnable demos for screen-share
├─ requirements.txt
├─ .env.example
└─ README.md
```

## What's new vs Session 29

| S29 (regex ReAct)                                | S30 (native function calling)                          |
|--------------------------------------------------|--------------------------------------------------------|
| `stop_sequences=["Observation:"]`                | Native `tools=[...]` parameter on the API call         |
| Regex parses `Action: tool[input]`               | Provider returns a structured `tool_use` block         |
| Single provider (Anthropic only)                 | Two providers behind one `Provider` interface           |
| One tool input string                            | Typed JSON arguments per tool schema                    |
| `print()` is your action log                     | `action_log.jsonl` — one structured line per call       |
| No parallel tool calls                           | Provider may emit multiple `tool_use` blocks per turn   |

## Running it

```bash
# one-time setup
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # then fill in your API keys

# run any of the demos
python demo.py 1                     # vague schemas → wrong tool every time
python demo.py 2                     # sharp schemas → right tool every time
python demo.py 3                     # multi-step research run (Anthropic)
python demo.py 4                     # same question against OpenAI

# or run all four
python demo.py
```

After every run, open `action_log.jsonl` — that's the production pattern this
session introduces. Every tool call leaves a structured trace: timestamp,
tool name, arguments, output, duration, status. This file is what an SRE
reads at 2am when something breaks.

## Tools shipped

| Tool                | What it does                                              |
|---------------------|-----------------------------------------------------------|
| `web_search`        | Look up a single public-web fact (stubbed for the demo)   |
| `calculator`        | Evaluate a single arithmetic expression                   |
| `wikipedia_summary` | Return a 1-2 paragraph encyclopedia summary of a topic    |

Both `web_search` and `wikipedia_summary` use deterministic offline stubs so
the demo runs in any classroom with no internet and no extra API keys. In
your portfolio build (the S30 exercise) you swap the stubs for real APIs —
SerpAPI / Tavily / Brave / the Wikipedia REST API. The schemas don't change.

## Portfolio handoff

This is the foundation for **Portfolio Project #4 — Tool-Calling Agent**.
Your exercise sheet for the session walks you through swapping the demo
research tools for tools specific to your Phase 5 domain, with sharp
schemas, action logging, and a 5-task evaluation run on real questions.

---
*Version v1.0 · Built May 2026 · `I Build. I Ship. I Teach.`*
