# Session 33 — Agentic RAG (Agent-Controlled Retrieval)

This folder is the reference implementation for Session 33 of the
Agentic AI Builders Fellowship. The session is about the discipline of
RETRIEVAL, not the surface area of TOOLS. Phase 4 taught you naive RAG;
Phase 5 puts an agent in charge of when, what, how many times, and
with what query the retrieval happens.

## What runs here

A tiny knowledge base for a fictional career-prep startup called
PrepDeck (six markdown documents in `corpus/`), loaded into a local
**ChromaDB** collection, with two retrieval surfaces:

- `naive_rag.py` — one embed, one retrieve, one generation. The
  baseline.
- `agentic_rag.py` — the agent loop. Decides WHEN to retrieve. Rewrites
  bad queries. Decomposes compound questions. Scores retrieved chunks
  through a quality gate. Re-queries when the gate fails. Stops when
  the retrieval budget runs out.

## PROD PATTERNS introduced this session

| Pattern | Where it lives | What it does |
|---|---|---|
| Retrieval as a Tool | `retrieval_tools.py`, `tools.py` | Hides the vector store behind a named tool with a sharp description. |
| **Retrieval Quality Gate** | `quality_gate.py` | Scores every retrieved chunk 1..5 before generation. Re-queries when avg < 3.5. The headline pattern. |
| Multi-Hop Retrieval | `agentic_rag.py` | On gate failure, the agent re-queries with a different formulation rather than answering from weak evidence. |
| Decompose-then-Retrieve | `query_rewriter.py`, `agentic_rag.py` | Compound questions are split into atomic sub-questions before retrieval. |
| Retrieval Budget | `agentic_rag.py` | A hard cap on `search_kb` calls per turn. Extends S31's Max-Step Ceiling to the retrieval surface. |

## Setup

```bash
cd Session_33/Code
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Set your Anthropic API key in a `.env` file next to this README:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Or run the demos against the offline fake LLM:

```bash
FAKE_LLM=1 python demo.py 1
```

## Ingest the corpus

```bash
python ingest.py            # builds ./chroma using the hash embedder
python ingest.py --inspect  # see what landed in the collection
```

The hash embedder is deterministic, offline, and good enough for the
demos. For the exercise you should switch to real embeddings:

```bash
USE_REAL_EMBEDDINGS=1 python ingest.py
```

(Requires `sentence-transformers`, installed by `requirements.txt`.)

## Run the demos

```bash
python demo.py 1     # WHEN — no-retrieval path on a maths question
python demo.py 2     # WHAT QUERY — naive vs agentic on a vague question
python demo.py 3     # DECOMPOSE — compound question, two retrievals
python demo.py 4     # QUALITY GATE — first retrieval fails, re-query wins
python demo.py 5     # BUDGET — adversarial question, ceiling hit
```

Every demo prints a labelled trace alongside the final answer. The
walkthrough script refers to the labels by name (e.g. "look for
`retrieve #1` and the `gate: FAIL` line just below it").

## File map

```
Session_33/Code/
  README.md                 ← you are here
  requirements.txt
  corpus/                   ← the 6 markdown docs that feed Chroma
    01_company_overview.md
    02_pricing.md
    03_engineering_handbook.md
    04_product_roadmap.md
    05_ai_stack.md
    06_hiring_faq.md

  rag_types.py              ← shared dataclasses
  embeddings.py             ← real + hash embedders behind one interface
  ingest.py                 ← chunk → embed → upsert into Chroma

  retrieval_tools.py        ← search_kb + list_documents + tool schemas
  tools.py                  ← single dispatcher for all tool calls

  llm_client.py             ← Anthropic wrapper + FAKE_LLM mode
  prompts.py                ← all system prompts in one place

  query_rewriter.py         ← rewrite() + decompose()
  quality_gate.py           ← chunk relevance scoring + verdict
  naive_rag.py              ← the baseline (one shot)
  agentic_rag.py            ← the agentic loop (the centre of the session)

  trace_logger.py           ← pretty-printers for the trace stream
  demo.py                   ← 5 walkthrough demos
```

## Callbacks to earlier sessions

- **S19** — ChromaDB. The collection format and the cosine-distance
  query interface are the same. We just hide them behind a tool.
- **S20** — RAG architecture. Naive RAG (this file's `naive_rag.py`) is
  exactly the S20 mental model.
- **S22–S23** — Retrieval techniques (filtering, MMR, hybrid). The
  `search_kb` tool exposes `source_filter` so the agent can pre-filter
  by document. Hybrid + rerank live one ring outside this session.
- **S28** — Where the 5 Levels of Agentic AI framework was introduced.
  Agentic RAG sits at **Level 3** (tool-using agent) for the simple
  WHEN-and-WHAT-QUERY demos and at **Level 4** (planning agent) when
  decomposition kicks in.
- **S30** — Tool design. The `search_kb` schema follows the SHARP
  description discipline from S30.
- **S31** — Max-Step Ceiling. The Retrieval Budget here is a domain-
  specific application of that pattern.
- **S32** — Context engineering. The Quality Gate is a context-
  engineering decision: don't let weak chunks into the synthesis prompt.

## Exercise

See `Session_33_Exercise.docx` (one folder up). The async exercise is
the curriculum's named build: convert your Phase 4 RAG pipeline into
an Agentic RAG system with chunk relevance scoring (1..5),
conditional re-querying on average < 3.5, and query rewriting on
re-retrieval. Run 10 test questions and submit the comparison.
