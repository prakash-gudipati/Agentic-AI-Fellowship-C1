# Session 38 — LangGraph: Stateful Agent Workflows

A research agent built as a **LangGraph state graph**. It analyses a question,
searches the real web, judges whether the results are good enough (a real LLM
quality gate — not a counter), and **loops back to search again with a refined
query** if they are not. When the results pass the gate, or the attempt ceiling
is hit, it writes a grounded report.

Two headline ideas, taught properly:

- **State + reducers** — one shared object (`ResearchState`) that every node
  reads and writes. Some boxes **overwrite** (latest score, current query);
  some boxes **accumulate** via a reducer (`Annotated[list, operator.add]`) so
  search hits and history GROW across loops instead of being erased. This is
  the idea beginners miss, and it is what makes a looping workflow remember.
- **The cycle** — a conditional edge after the gate routes either back to
  `search` (loop) or on to `write_report` (exit). Chains can't do this.

```
START -> analyze_query -> search -> evaluate -> {search (loop) | write_report} -> END
```

Then a second demo shows the **prebuilt path**: assemble a real ReAct
tool-calling agent from `MessagesState` + `ToolNode` + `tools_condition` — the
same decide/act/observe loop from S29, in ~12 lines. (The one-liner
`langchain.agents.create_agent` builds this exact graph; we build it from the
pieces so you can see there's no magic.)

## This session uses REAL services

No offline fake mode for the demo — it calls the real Anthropic API and runs
real web searches. To verify the **graph wiring** (state accumulation, loop,
brake, prebuilt assembly) without any key or network, use `--selftest`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY (and optionally TAVILY_API_KEY)
```

No Tavily key? Search falls back to keyless DuckDuckGo automatically.

## Run it

```bash
python demo.py --graph            # draw the hand-wired research graph (no key)
python demo.py --prebuilt-graph   # draw the prebuilt ReAct agent graph (no key)
python demo.py --selftest         # prove state/loop/brake/prebuilt wiring (no key)

python demo.py "What is LangGraph and how does it differ from LangChain?"
python demo.py --prebuilt "Who won the most recent Cricket World Cup final?"
```

The hand-wired run prints a live trace: `SEARCH +4 new (4 total)`, then
`GATE 0.45 -> FAIL`, then `LOOP`, then a second `SEARCH +4 new (8 total)` —
visible proof the reducer is accumulating — then `GATE 0.82 -> PASS`, report.

## Files

| File | Role |
|------|------|
| `state.py` | `ResearchState` — overwrite boxes vs **accumulate (reducer) boxes** |
| `prompts.py` | Three system prompts (analyzer, evaluator/gate, reporter) |
| `llm.py` | The one shared `ChatAnthropic` model + three call helpers |
| `search_tools.py` | Real web search — Tavily primary, DuckDuckGo keyless fallback |
| `nodes.py` | The four node functions (return only changed keys; reducers append) |
| `router.py` | `should_continue` — the **pure** conditional-edge function (the loop) |
| `graph.py` | Wires the hand-built research graph and compiles it |
| `prebuilt_agent.py` | The prebuilt path: `MessagesState` + `ToolNode` + `tools_condition` |
| `trace.py` | Console trace printers for the live walkthrough |
| `selftest.py` | Offline wiring tests (state/loop/reducer/brake/prebuilt) |
| `demo.py` | CLI: `--graph`, `--prebuilt-graph`, `--selftest`, real / `--prebuilt` runs |

## PROD PATTERNS

- **Quality-Gated Cycle** (headline) — loop on a *real* quality check, with a
  hard attempt ceiling so the loop always terminates.
- **State as a typed contract with reducers** — overwrite for current values,
  accumulate (`operator.add` / `add_messages`) for things that grow.
- **Prebuilt over hand-rolled when it fits** — `ToolNode` + `tools_condition`
  (or `create_agent`) replace boilerplate once you know what they wire.

## Callbacks

S29 (the hand-rolled ReAct loop the prebuilt agent now packages), S31
(Max-Step Ceiling -> attempt brake), S33 (Retrieval Quality Gate -> the gate
edge), S37 (LangChain chains only flow forward -> a graph loops; same
`.invoke()`/`.stream()`).
