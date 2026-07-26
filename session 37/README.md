# Session 37 — LangChain (combined): Intro + Hands-On

Reference code for the combined LangChain session. Walkthrough format —
the instructor explains and runs this code on screen share; nobody types
it live.

## What this demonstrates

| Demo | Curriculum point | What you see |
|------|------------------|--------------|
| 1 | LangChain vs pure Python | Same summarisation task, two code shapes. What the framework does for free — and what it hides. |
| 2 | Chains, prompt templates, output parsers, LCEL | `prompt \| model \| parser` and variable substitution. |
| 3 | Document Q&A chain | Retrieve → stuff → answer, as one LCEL chain over the NovaDesk docs. |
| 4 | Summarisation + streaming | The same three-link chain, but `.stream()` yields tokens as they arrive. |
| 5 | Callbacks, caching, debugging | A callback handler logs every step; an in-memory cache serves a repeat call; a wrong-answer trace shows how to debug. |

## Run it

```bash
pip install -r requirements.txt

# Offline, deterministic, no API key (what the instructor runs pre-class):
PYTHONPYCACHEPREFIX=/tmp/s37_pycache FAKE_LLM=1 python demo.py all

# One demo at a time:
FAKE_LLM=1 python demo.py 1     # ... through 5

# Real Claude calls (needs ANTHROPIC_API_KEY + langchain-anthropic):
python demo.py 3
```

## Files

```
corpus.py               NovaDesk docs (stands in for a document loader)
model_factory.py        get_chat_model() -> real ChatAnthropic OR RoutedFakeChatModel
prompts.py              system prompts (each opens "You are a <role>.")
retriever.py            keyword retriever + RecursiveCharacterTextSplitter, as a Runnable
chains.py               the three LCEL chains (template / summary / Q&A)
callbacks.py            StepLoggingCallbackHandler — PROD PATTERN: observability
pure_python_baseline.py the 'before' picture for Demo 1 (raw Anthropic SDK)
trace_logger.py         ANSI printers
demo.py                 the five demos + CLI
```

## How offline mode works

`FAKE_LLM=1` swaps `ChatAnthropic` for `RoutedFakeChatModel`, a real
`BaseChatModel` subclass that returns deterministic canned answers by
reading the system prompt's opener phrase. The LangChain wiring — prompts,
parsers, LCEL pipes, retrieval, streaming, callbacks, caching — is all
REAL. Only the model's replies are faked. That is why the traces look
identical to a real run.

## Production patterns introduced

1. **Framework Judgement** — use LangChain when it saves time, not by default.
2. **LCEL Composition** — build pipelines as `a | b | c` of Runnables.
3. **Callback-Based Observability** — log every chain step (the by-hand version of S26's hosted tracing platform).
4. **Response Caching** — an identical call is served from cache, not re-billed.
