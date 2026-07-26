# Session 35 — Multi-Agent Systems II (Advanced Architectures)

Reference code for the four advanced multi-agent patterns introduced
in S35:

1. **Hierarchical Manager-of-Managers** (headline)
   `Director → ResearchManager + EditorialManager → 3 leaf workers`
2. **Debate / Consensus**
   `Bull + Bear + Neutral panelists → Moderator synthesises`
3. **Competitive / Best-of-N**
   `N CandidateWriters in parallel → Judge picks a winner`
4. **Communication Protocol** (cross-cutting)
   Typed `Message` bus carries every cross-agent exchange.

## Run the demos (offline, no API key)

```bash
cd Session_35/Code
PYTHONPYCACHEPREFIX=/tmp/s35_pycache FAKE_LLM=1 python demo.py 1   # hierarchical
PYTHONPYCACHEPREFIX=/tmp/s35_pycache FAKE_LLM=1 python demo.py 2   # debate
PYTHONPYCACHEPREFIX=/tmp/s35_pycache FAKE_LLM=1 python demo.py 3   # competitive
PYTHONPYCACHEPREFIX=/tmp/s35_pycache FAKE_LLM=1 python demo.py 4   # protocol replay
PYTHONPYCACHEPREFIX=/tmp/s35_pycache FAKE_LLM=1 python demo.py 5   # escalation
PYTHONPYCACHEPREFIX=/tmp/s35_pycache FAKE_LLM=1 python demo.py all # all five
```

## Run the demos with a real Anthropic key

```bash
cp .env.example .env
# edit ANTHROPIC_API_KEY in .env
python demo.py all
```

## File map

| File | Lines | Purpose |
|------|-------|---------|
| `agent_types.py` | ~200 | Dataclasses: Message, Fact, Argument, ConsensusReport, JudgeVerdict, results |
| `messages.py` | ~165 | MessageBus — typed append-only inter-agent bus |
| `prompts.py` | ~175 | System prompts for every role (opener-phrase starts) |
| `llm_client.py` | ~625 | Anthropic wrapper + FAKE_LLM with opener-phrase router |
| `trace_logger.py` | ~185 | ANSI-coloured `handle_event` printer |
| `tools.py` | ~24 | Stub dispatcher (placeholder — most S35 agents don't need tools) |
| `agents/director.py` | ~180 | Top-level orchestrator — never talks to workers |
| `agents/research_manager.py` | ~165 | Splits, dispatches researchers in parallel, merges |
| `agents/editorial_manager.py` | ~145 | Writer-FactChecker loop with revision bound |
| `agents/researcher.py` | ~65 | Leaf worker — facts only |
| `agents/writer.py` | ~45 | Leaf worker — prose only |
| `agents/fact_checker.py` | ~60 | Leaf worker — verdict only |
| `agents/debate_panel.py` | ~200 | Bull / Bear / Neutral + Moderator |
| `agents/competitive_panel.py` | ~155 | N CandidateWriters + Judge |
| `demo.py` | ~315 | 5 demos + CLI dispatcher + cost math |

Total: **~2,700 LOC**.

## Phase 5 build conventions in use

1. **Opener-phrase routing.** Every system prompt in `prompts.py` starts
   with `"You are a <role>."`. The `_fake_one_shot()` dispatcher in
   `llm_client.py` routes on those opener phrases — never on a topic
   word. See `CLAUDE.md` PHASE 5 BUILD CONVENTIONS for the why.

2. **PYCACHE escape hatch.** Set `PYTHONPYCACHEPREFIX=/tmp/s35_pycache`
   if your filesystem won't let Python delete `__pycache__/`. Without
   it, stale bytecode survives source edits.

3. **Walkthrough-only format.** No live build. Instructor opens these
   files on screen share, runs demos in the terminal, and explains the
   trace.

## PROD PATTERNS introduced

| Pattern | Where it lives | Callback |
|---------|----------------|----------|
| **Hierarchical Decomposition** (headline) | `Director` + Managers | Extends S34 Orchestrator-Worker Decomposition one level up |
| **Inter-Agent Message Schema** | `messages.py` + `agent_types.Message` | Extends S34 Shared Scratchpad to a typed bus |
| **Consensus Synthesis** | `agents/debate_panel.py` | New shape — not in S34 |
| **Best-of-N Judging** | `agents/competitive_panel.py` | New shape — sibling of S34 Critique Loop, different output |
