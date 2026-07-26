# Session 39 — LangGraph Advanced: Persistence, HITL + Long-Term Memory

**Portfolio Project #6.** This is the Human-in-the-Loop Research Agent. It takes
the Session 38 Research Workflow (analyze → search → evaluate → loop → report)
and adds the four capabilities that turn a one-shot graph into a production
agent you can trust:

| Capability | What it adds | The mechanism |
|---|---|---|
| **Persistence** | A run survives a program restart and resumes mid-flight | a **checkpointer** (`SqliteSaver`) saves State after every node |
| **Human-in-the-Loop** | The agent pauses, shows its plan, waits for a human | `interrupt()` + `Command(resume=...)` |
| **Long-term memory** | Knowledge carries from one run to a *different* run | the **Store** (`InMemoryStore`), cross-thread |
| **Time-travel** | Rewind to any past checkpoint and fork an alternate run | `get_state_history()` + `update_state()` |

Plus a `RetryPolicy` on the network node so a transient search failure retries
instead of crashing the run.

## The two memories — do not confuse them

- **Checkpointer = per-thread, short-term.** One `thread_id` = one conversation.
  It is what makes a single run pausable/resumable. Like a video-game autosave.
- **Store = cross-thread, long-term.** Shared across every run. Thread A writes,
  thread B (next week) reads. Like the game's shared unlocked-achievements file.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY for a real run

# --- offline (no key, no network): set FAKE_LLM=1 ---
FAKE_LLM=1 python demo.py --selftest        # 16 wiring checks
FAKE_LLM=1 python demo.py --graph           # draw the workflow
FAKE_LLM=1 python demo.py --restart "..."   # state survives a restart
FAKE_LLM=1 python demo.py --store           # cross-thread memory
FAKE_LLM=1 python demo.py --time-travel "..."  # rewind + fork

# --- real run (needs ANTHROPIC_API_KEY) ---
python demo.py --hitl "what is retrieval-augmented generation"
# pauses, shows the plan; press Enter to approve or type an edited plan
```

## Files

```
state.py         the State, extended with plan / approved_plan / prior_context
prompts.py       4 system prompts (planner is new); opener-phrase routing
llm.py           ChatAnthropic + offline FAKE brain
search_tools.py  Tavily -> DuckDuckGo -> offline canned corpus
persistence.py   build the checkpointer (Sqlite/in-memory) and the Store
nodes.py         recall / plan / human_review(interrupt) / ... / save
router.py        should_continue — the pure cycle decision (from S38)
graph.py         wiring + compile(checkpointer=, store=) + RetryPolicy
trace.py         labelled console trace for the walkthrough
selftest.py      16 offline wiring checks (the pre-session safety net)
demo.py          --graph / --selftest / --hitl / --restart / --store / --time-travel
```

## Smoke-test commands (instructor pre-session checklist)

```bash
export PYTHONPYCACHEPREFIX=/tmp/s39_pycache FAKE_LLM=1
python demo.py --selftest          # expect 16/16
python demo.py --restart "what is agent memory"
python demo.py --store
python demo.py --time-travel "what is chunking"
echo approve | python demo.py --hitl "what is RAG"
```

## What's deliberately left as a one-line mention

`Send` (dynamic parallel fan-out) and `Command` (combined state-update + routing
in a single return) are real LangGraph features but belong to a parallelism
session, not this one. They are named in the slides as "here's what exists" and
not built here — the four capabilities above are already a full portfolio piece.

## Production notes

- The `InMemoryStore` shown here has the **same API** as the production
  Postgres-backed store; swapping it is a deployment change, not a code change.
- `interrupt()` is the modern HITL primitive. `interrupt_before=["node"]` at
  compile time is the older static-breakpoint approach — same idea, less
  flexible. Both pause the graph; this project uses the dynamic `interrupt()`.
- Tracing is hard-disabled at startup (this is not the observability session).
