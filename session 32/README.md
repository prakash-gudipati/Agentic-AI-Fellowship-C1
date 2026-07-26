# Session 32 — Agent Memory Systems + Context Engineering for Agents

Reference code for Phase 5, Session 32 of the Agentic AI Builders
Fellowship. Walkthrough-only format — the instructor runs the demos on
screen share while explaining the architecture.

## What's inside

```
Session_32/Code/
  memory_types.py        # Turn, SummaryNote, SemanticHit, EvictionEvent, SessionStateSnapshot
  working_memory.py      # Token-bounded verbatim buffer (the SHORT-TERM layer)
  eviction.py            # Recency / Importance / Hybrid policies (PROD PATTERN)
  summariser.py          # Rolling summary buffer                  (PROD PATTERN)
  semantic_memory.py     # Tiny vector index for long-term recall  (PROD PATTERN)
  session_state.py       # Save / load JSON artifact               (PROD PATTERN)
  memory_store.py        # SQLite-backed durable substrate         (production swap)
  context_builder.py     # Assembles {system, summary, hits, working, user}
  llm_client.py          # Anthropic wrapper with FAKE_LLM=1 path
  agent.py               # MemoryAgent — integrates every layer
  tools.py               # calculator + note_keeper (kept tiny on purpose)
  trace_logger.py        # Terminal pretty-printer for the walkthrough
  demo.py                # 6 demos
  requirements.txt
```

## The four PROD PATTERNS

| Pattern | Module | Purpose |
| --- | --- | --- |
| Conversation Summarisation | `summariser.py` | Roll evicted turns into a running summary so context survives eviction |
| Memory Eviction Policy | `eviction.py` | Named, configurable rule for who leaves working memory |
| Semantic Memory via Vector Recall | `semantic_memory.py` | Embed evicted turns; surface them back when relevant |
| Session State as Artifact | `session_state.py` (JSON) / `memory_store.py` (SQLite) | Persist every memory layer for save / replay |

## Running the demos

### Real API mode (default)

```bash
cd Session_32/Code
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
pip install -r requirements.txt
python demo.py 3
```

### Offline mode (no API key)

```bash
FAKE_LLM=1 python demo.py 3
```

`FAKE_LLM=1` is enough to demonstrate every pattern. The canned replies
in `llm_client.py` are deterministic, so the walkthrough produces the
same output every time.

## The demos

1. **Amnesia baseline** — no memory, agent forgets after each turn.
2. **Working memory only** — verbatim sliding window; works until it overflows.
3. **Summarisation** — rolling summary catches the overflow.
4. **Semantic recall** — pulls back a turn that left the buffer many turns ago.
5. **Session state as JSON artifact** — save to JSON, rehydrate a fresh agent.
6. **Production substrate — SQLite** — same four patterns, but every memory
   mutation is mirrored to a single SQLite file. Process A writes; Process B
   opens the same file and answers from the DB.

## Production substrate — SQLite

Demos 1–5 keep every memory layer in Python RAM. That's the right teaching
shape — you see the patterns without the noise of a DB. Demo 6 swaps RAM
for one SQLite file via `memory_store.MemoryStore`.

When you pass `store=MemoryStore.open("memory.db")` to `MemoryAgent`,
every state change is mirrored to the DB:

- `turns` table — every Turn ever appended, with an `in_working` flag
- `summaries` table — UPSERTed on every fold
- `vectors` table — one row per evicted turn, vector stored as JSON
- `eviction_log` table — one row per eviction event
- `sessions` table — one row per agent session

Semantic recall is computed in Python over the candidates fetched from the
`vectors` table. This is honest about the scale ceiling:

- **Works** up to ~10,000 vectors per session, ~50 ms query time.
- **Breaks** past that. Swap to `sqlite-vec`, ChromaDB, pgvector, or a
  dedicated vector store (Pinecone / Weaviate / Qdrant).

```python
from memory_store import MemoryStore
from agent import MemoryAgent

store = MemoryStore.open("memory.db")

# Process A — write a session
agent_a = MemoryAgent(store=store, working_budget=120)
agent_a.chat("My name is Aanya.")
agent_a.chat("I'm building PrepDeck.")
session_id = agent_a.session_id
store.close()

# Process B — fresh handle on the same file
store_b = MemoryStore.open("memory.db")
agent_b = MemoryAgent.from_store(store_b, session_id=session_id)
agent_b.chat("Remind me my name.")  # answers from the DB
```

### Filesystem caveat

SQLite needs POSIX file locking. On Windows-share / FUSE / some networked
mounts the locking dance fails with `disk I/O error`. Set
`S32_DB_PATH=/some/local/path/memory.db` to override the default path,
or use `pragma journal_mode = MEMORY` (already on by default in this build).

## How this connects to earlier sessions

- The semantic-memory layer is your Phase 4 RAG stack (S19, S20-S22)
  pointed at the agent's own conversation history instead of a document
  corpus. Same machine. New corpus.
- The session-state artifact extends S31's "Plan as Artifact" pattern
  from a single plan object to the entire memory state.
- The cost ledger in `trace_logger.print_context_ledger` follows
  the discipline introduced in S26 (observability + cost management).
- The agent loop is the smallest possible loop — S29 / S30 cover the
  full ReAct + native-function-calling versions.

## Memory ≠ context engineering

Memory is the MECHANISM — what gets stored, where, and how.

Context engineering is the DISCIPLINE — what goes into the model prompt
this turn, in what order, under what budget.

`context_builder.py` is where the discipline lives. Every decision is
visible, deterministic, and printable. That is the entire point.
