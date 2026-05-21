# Session 29 — ReAct Agent from Scratch

Pure Python. Anthropic SDK only. No frameworks.

## What this is

A minimal but production-flavoured implementation of the **ReAct** pattern —
Reasoning **+** Acting. The agent thinks one step at a time, calls a tool,
reads the tool's output, then thinks again, until it has enough information
to answer the original question.

## Files

| File | Role |
|---|---|
| `agent.py` | The ReAct loop. The one file students should read end-to-end. |
| `tools.py` | Three tools: `calculator`, `current_datetime`, `web_search` (stubbed). |
| `prompts.py` | The system prompt that teaches the LLM the ReAct format. |
| `parser.py` | Pulls `Action: tool[input]` and `Final Answer:` lines out of LLM output. |
| `retry.py` | The S29 production pattern — exponential-backoff retry around tool calls. |
| `trace_logger.py` | Pretty-printer for the Thought → Action → Observation trace. |
| `demo.py` | Three runnable example questions. The walkthrough script. |

## Run it

```bash
cp .env.example .env             # then fill in your ANTHROPIC_API_KEY
pip install -r requirements.txt
python demo.py                   # runs all three demo questions
python demo.py 2                 # runs only question 2
```

## What students should notice

1. **The agent file is short.** Everything else is plumbing. The loop *is*
   the concept.
2. **Stop sequences make the loop reliable.** The LLM stops at
   `Observation:`, we inject the real observation, the LLM continues. No
   "did the model hallucinate the tool output?" failure mode.
3. **Retry is a production concern, not an LLM concern.** When a tool fails,
   the *agent* retries — the LLM never sees the flakiness.
4. **No frameworks.** Every line is one we can explain. That's the point of
   doing this before LangChain (S37).
