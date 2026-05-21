# Session 25 — Production RAG Evaluation — Ragas Deep Dive

This is the reference implementation for Session 25 of the Agentic AI
Builders Fellowship. It takes the RAG pipeline you built in S20–S24 and
makes it **measurable** — with Ragas as the named evaluation framework
and the production scaffolding around it.

## What's new in S25

1. **Embedding cache** — `embedding_cache.py` — disk-backed
   `(model, sha256(text)) → vector` cache. Re-runs of the same corpus
   pay zero.
2. **Ragas DEEP** — `evals/harness_ragas.py` — all 5 Ragas metrics:
   faithfulness, answer relevancy, context precision, context recall,
   plus **AspectCritic** (the escape-hatch for custom rubrics). Judge
   model is configurable; cost is tracked; one full judge prompt is
   printed so you can SEE what an LLM-as-judge call looks like.
3. **Diagnostic reading** — `evals/diagnostic.py` — reads a Ragas
   score dict and emits the weakest layer, the most likely root cause,
   and a specific fix. The "score → fix" decision tree.
4. **Synthetic test data** — `evals/synthetic_dataset.py` — wraps
   Ragas's `TestsetGenerator` (or an offline fallback) so you can
   bootstrap an eval dataset from your corpus in minutes.
5. **Score log** — `evals/score_log.py` — append-only CSV stamped with
   timestamp + git commit hash. The regression log.

Everything else (loaders, chunker, embeddings, similarity retriever)
is carried over from S24 unchanged. The retriever strategy pattern
keeps the contract identical.

> **DeepEval lives in S41 and S42.** Its safety metrics
> (HallucinationMetric, BiasMetric, ToxicityMetric) and workflow
> features (BaseMetric subclassing + pytest integration) are taught
> when they land naturally — in the Guardrails + Evals and
> Eval-Driven Development sessions.

## Layout

```
Code/
├── data/                       # demo corpus (3 docs)
├── loaders.py                  # carried from S22-S24
├── chunker.py                  # carried from S22-S24
├── embeddings.py               # carried from S22-S24
├── embedding_cache.py          # NEW — content-hash cache
├── retrievers/
│   ├── base.py                 # carried — ChunkIndex + Retriever
│   └── similarity.py           # carried — cosine retriever
├── pipeline.py                 # NEW — question → answer callable
├── evals/
│   ├── golden_dataset.json     # 10-row source of truth
│   ├── golden_dataset.py       # loader
│   ├── harness_custom.py       # hand-rolled baseline
│   ├── harness_ragas.py        # all 5 Ragas metrics
│   ├── synthetic_dataset.py    # TestsetGenerator wrapper
│   ├── diagnostic.py           # score-shape → fix
│   └── score_log.py            # regression CSV
├── semantic_cache.py           # NEW — the "other" cache pattern
├── demo.py                     # signature 6-act demo
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

python demo.py
```

The demo runs in six acts:

| Act | What happens | Why it's there |
|-----|--------------|---------------|
| 1 | Build the index with a COLD cache | Every chunk is a miss → real API calls |
| 2 | Rebuild the index with a WARM cache | Same texts → all hits → free |
| 3 | Ragas DEEP — 5 metrics with one judge prompt printed verbatim | See what an LLM-as-judge call actually looks like |
| 4 | Diagnostic — score shape → recommended fix | The most-shipped artifact of this session |
| 5 | Synthetic test-data preview | Bootstrap eval rows from a corpus in minutes |
| 6 | Append today's row to `score_log.csv` | The regression log |

### Faster paths

```bash
python demo.py --skip-evals          # acts 1-2 + 5 only (no LLM-judge calls)
python demo.py --skip-synthetic      # acts 1-4 + 6 (no LLM bootstrap)
```

## Offline mode

`harness_ragas.py` and `synthetic_dataset.py` both fall back to a tiny
in-house LLM-as-judge implementation if the `ragas` package isn't
installed. Scores will not be byte-identical to real Ragas, but the
shape of the output is the same — so the rest of the demo (and your
diagnostic reading) works on a plane.

## Plugging in your own corpus

Drop your `.pdf`, `.txt`, `.md`, or `.html` files into `data/`, then
edit `DEFAULT_CORPUS` at the top of `demo.py` to point at them. You'll
also want to rebuild `evals/golden_dataset.json` with 10 questions
specific to your corpus — the dataset IS the eval.
