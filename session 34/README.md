# Session 34 — Multi-Agent Systems I: Patterns

Reference code for Session 34 of the Agentic AI Builders Fellowship.
The session moves from a single agent (S29–S33) to a coordinated CREW
of specialist agents.

## What runs here

Four multi-agent patterns demoed across five scenarios:

| Demo | Pattern | What happens |
|------|---------|---|
| 1 | Single vs Crew | Same question to a single-agent baseline AND a 3-worker crew. Crew wins on output quality, pays in LLM calls. |
| 2 | Sequential Pipeline | Researcher → Writer. No orchestrator. Fixed order. |
| 3 | Parallel / Map | 3 researchers fan out on 3 frameworks; aggregator merges. |
| 4 | Critique Loop | Writer plants a bad claim, Fact-Checker catches it, Writer revises, Fact-Checker accepts. |
| 5 | Termination ceiling | Adversarial question; Fact-Checker never accepts; max_rounds=3 stops the loop. |

## PROD PATTERNS introduced

| Pattern | Where it lives | What it does |
|---|---|---|
| Role Specialization | `prompts.py`, `agents/*.py` | One role per agent. Narrow prompt + narrow output shape. |
| Shared Scratchpad | `scratchpad.py` | Typed sections (FACTS / DRAFT / CRITIQUE / FINAL). Not group chat. |
| Termination Conditions | `agents/orchestrator.py` | max_rounds, all_done, quality_met. |
| **Orchestrator-Worker Decomposition** | `agents/orchestrator.py` | The headline. Orchestrator decomposes, never generates. |

## Setup

```bash
cd Session_34/Code
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Set your API key in `.env` (copy from `.env.example`) or use offline mode.

## Run the demos

```bash
# Recommended — Phase 5 build convention env vars baked in:
PYTHONPYCACHEPREFIX=/tmp/s34_pycache FAKE_LLM=1 python demo.py 1
```

```bash
python demo.py 1     # Single vs Crew
python demo.py 2     # Sequential pipeline
python demo.py 3     # Parallel / Map
python demo.py 4     # Critique Loop (writer + fact-checker)
python demo.py 5     # Termination ceiling on adversarial Q
```

## File map

```
Session_34/Code/
  README.md                    ← you are here
  requirements.txt             ← anthropic only
  .env.example
  agent_types.py               ← Task / Fact / Critique / CrewResult
  scratchpad.py                ← typed shared memory + sections
  prompts.py                   ← 5 system prompts (every one starts "You are a <role>")
  llm_client.py                ← Anthropic wrapper + FAKE_LLM router (opener-phrase routes)
  trace_logger.py              ← ANSI-coloured event printers
  single_agent_baseline.py     ← the baseline crew beats in Demo 1
  agents/
    __init__.py
    researcher.py              ← gathers facts
    writer.py                  ← composes prose
    fact_checker.py            ← reviews against facts
    orchestrator.py            ← centre of the session — decompose, dispatch, terminate
  demo.py                      ← 5 demos
```

## Callbacks to earlier sessions

- **S29** — ReAct loop. The agents here are essentially S29 loops with narrower prompts.
- **S30** — Tool design. Each agent's tool subset is the same SHARP-description discipline applied at agent granularity.
- **S31** — Max-Step Ceiling → Termination Conditions (max_rounds).
- **S32** — Context Engineering → Shared Scratchpad (typed sections vs group chat).
- **S33** — Retrieval Quality Gate → Critique Loop (same shape, different decision).

## Exercise

See `Session_34_Exercise.docx` one folder up. Wrap your existing single-agent code (S29 or S30) in an orchestrator + 1 specialist worker. Add typed scratchpad + max_rounds. Run 8 test questions. 2-week build window.
