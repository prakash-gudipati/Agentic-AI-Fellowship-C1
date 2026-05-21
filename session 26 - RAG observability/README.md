# Session 26 — Observability — Tracing, Debugging + Cost Management

This is the reference implementation for Session 26 of the Agentic AI
Builders Fellowship. It takes the S25 RAG-with-evals pipeline and adds
**observability**: every query produces a LangSmith trace, every LLM
call is auto-instrumented for token + cost, and a CI gate fails the
build when projected daily spend exceeds an agreed threshold.

The session anchors on **LangSmith** — the hosted observability + eval
platform from the LangChain team — because it's the single most-named
observability tool in 2025–2026 AI engineer job postings. Opik (the
open-source self-hosted alternative) is named so students know both.

## What's new in S26

1. **`@traceable` decorator** (in `observability/langsmith_tracing.py`)
   — one line above a function makes every call a recorded LangSmith
   run. The decorated function's body does not change.
2. **Auto-instrumented LLM client** (`wrap_llm_client`) — wraps the
   Anthropic SDK so every `messages.create(...)` becomes a child run
   with token counts + dollar cost attached automatically. No tiktoken,
   no custom pricing table.
3. **Trace replay debugger** (`observability/debugger.py`) — given a
   failed `run_id`, pull the original input from LangSmith, re-run
   locally with a different model or retriever, print a side-by-side
   diff of latency / cost / answer.
4. **$/day budget gate** (`observability/cost_budget.py`) — query the
   LangSmith API for actual run cost over the last N runs, project to
   daily spend, fail CI if it exceeds the threshold. Same shape as
   S25's eval gate; different number.
5. **Offline JSONL fallback** — every observability path works without
   a LangSmith API key. Traces append to `.langsmith_traces.jsonl`
   with the same shape LangSmith uses. The lesson runs on a plane.

Everything else (loaders, chunker, embeddings, retrievers, evals,
golden dataset, semantic cache) is carried over from S25 unchanged.

## What's NOT in S26 — by design

We deliberately do NOT write:

- A custom **CostMeter** — LangSmith captures cost from the LLM
  response automatically once `wrap_anthropic()` is applied. Building
  our own meter would duplicate the platform.
- A custom **dashboard** — LangSmith's hosted UI already aggregates
  latency, cost, and error rate over time. We point students at the
  UI instead of writing terminal renderers.

This is the production rule named today: **use the platform, not
custom code.** Anything LangSmith does for free, we don't rebuild.

## Layout

```
Code/
├── data/                                 # demo corpus (3 docs)
├── loaders.py                            # carried from S20-S25
├── chunker.py                            # carried from S20-S25
├── embeddings.py                         # carried from S20-S25
├── embedding_cache.py                    # carried from S25
├── semantic_cache.py                     # carried from S25
├── retrievers/
│   ├── base.py                           # carried — ChunkIndex + Retriever
│   └── similarity.py                     # carried — cosine retriever
├── pipeline.py                           # UPDATED — @traceable + wrap_llm_client
├── evals/                                # carried from S25 (Ragas DEEP, golden, diag, etc.)
├── observability/                        # NEW
│   ├── __init__.py                       # package exports
│   ├── langsmith_tracing.py              # @track decorator + wrap_llm_client + offline fallback
│   ├── cost_budget.py                    # CI cost gate (LangSmith API + offline fallback)
│   └── debugger.py                       # trace replay (pull-by-run_id + offline fallback)
├── demo.py                               # 5-act signature demo
├── requirements.txt
└── .env.example
```

## Run it

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# put your real API keys in .env
#   - ANTHROPIC_API_KEY (required)
#   - OPENAI_API_KEY (for the Ragas embedding judge — carried from S25)
#   - LANGCHAIN_API_KEY (optional — leave blank for offline mode)

python demo.py
```

The demo runs in five acts:

| Act | What happens | Why it's there |
|-----|--------------|---------------|
| 1 | Build the index with a COLD cache (traced) | Every embedding call is a miss → real API hits, every miss is a span |
| 2 | Rebuild the index with a WARM cache (traced) | Same texts → all hits → free; the trace shows the cache path |
| 3 | Run the S25 golden-dataset eval with tracing on every row | Each row is one LangSmith trace; URL printed for live UI inspection |
| 4 | Find the lowest-faithfulness row, REPLAY with a stronger model | Side-by-side diff of original vs replay; "fix it without merging" |
| 5 | $/day budget gate — pass + fail demos | Pulls real cost from LangSmith (or offline log); fails the build at threshold |

### Faster paths

```bash
python demo.py --skip-evals          # acts 1-2 only — no LLM-judge calls
python demo.py --daily-budget 0.50   # tight budget so the FAIL tier triggers fast
python demo.py --replay-model claude-sonnet-4-6
```

## The two modes — LangSmith on vs LangSmith off

| Path | `LANGCHAIN_API_KEY` set? | Tracing source | Cost source |
|------|--------------------------|----------------|-------------|
| Production | yes | hosted LangSmith UI | LangSmith Run API |
| Offline    | no  | `.langsmith_traces.jsonl` | same JSONL, parsed locally |

Both paths use the exact same code call sites. The decorator dispatches
internally. This is the **graceful degradation** pattern named today.

## Plugging in your own corpus

Drop your `.pdf`, `.txt`, `.md`, or `.html` files into `data/`, then
edit `DEFAULT_CORPUS` at the top of `demo.py` to point at them. The
golden dataset (`evals/golden_dataset.json`) is corpus-specific — you
will want to rebuild it (or run `synthetic_dataset.py`) with 10
questions specific to your new corpus.

## The capstone is next

Session 27 is the **RAG Capstone — Production-Grade Pipeline**. The S25
+ S26 stack you have here is the foundation: chunking, retrieval, eval
gates, tracing, cost budget. The capstone wires three or more retrieval
techniques together behind the same scaffolding and asks you to defend
your pick with eval numbers.

Read `Session_27_Capstone_Brief.docx` (linked in the chat) when you're
ready to start the 2-week build.
