# Session 40 — Guardrails + Evals: a Production Safety Layer

Reference code for a customer-support agent (fictional fintech **PayMint**)
wrapped in a two-part safety layer:

1. **Guardrails** — runtime, per-request defense. Input checks run *before* the
   model; output checks run *after* it, before the user sees anything.
2. **Eval harness** — systematic batch measurement of quality + safety with
   **DeepEval**, run like a test suite against a golden dataset.

Everything runs **fully offline** under `FAKE_LLM=1` — no API key, no cost,
deterministic output. Only the model's replies are faked; the DeepEval metric
machinery (GEval, AnswerRelevancyMetric, LLMTestCase) is the real thing.

## Run all demos offline

```bash
pip install -r requirements.txt --break-system-packages
PYTHONPYCACHEPREFIX=/tmp/s40_pycache DEEPEVAL_TELEMETRY_OPT_OUT=YES FAKE_LLM=1 python demo.py all
```

Run a single demo: `FAKE_LLM=1 python demo.py 4` (1..5 or `all`).

## Environment variables

| Var | Purpose |
|---|---|
| `FAKE_LLM=1` | Offline canned model. Default-on when no key is present. |
| `ANTHROPIC_API_KEY` | Real path (Claude Haiku). Only used when `FAKE_LLM != 1`. |
| `DEEPEVAL_TELEMETRY_OPT_OUT=YES` | Silences DeepEval's telemetry HTTP call. |
| `NO_COLOR=1` | Disable ANSI colours in the trace. |
| `PYTHONPYCACHEPREFIX=/tmp/...` | Keep `.pyc` off the mount during dev. |

## The five demos

1. **Input Guardrail Chain** — one input with an injection phrase AND an email:
   PII is redacted first, then injection BLOCKs (first-BLOCK short-circuit).
2. **Output Validation Gate** — a model answer that leaks an email and a
   hallucinated number: PII redacted, then GroundingGuardrail BLOCKs.
3. **Fail-Closed Default** — a guardrail that raises; same input BLOCKs when
   `fail_closed=True`, passes when `False`.
4. **SafeAgent end-to-end** — a safe question passes and calls the model; an
   unsafe question is blocked *before* the model (`model_called=False`).
5. **Eval Harness** — the 10-case golden dataset scored with DeepEval metrics +
   a safety-coverage assertion, printed as a scorecard with an overall gate.

## Files

| File | What it is |
|---|---|
| `guardrail_types.py` | Decision/Severity enums + result/report dataclasses + cost helpers |
| `prompts.py` | Agent + two LLM-as-judge system prompts (opener-phrase routed) |
| `llm_client.py` | `complete()` — real Anthropic path + offline FAKE router |
| `input_guardrails.py` | Injection / PII / Length / Topic input checks |
| `output_guardrails.py` | PII-redaction / Schema / Safety-judge / Grounding-judge output checks |
| `guardrail_chain.py` | Composable chain — first-BLOCK short-circuit + **fail-closed default** |
| `safe_agent.py` | `SafeAgent.ask()` — input gate → model → output gate |
| `eval_dataset.py` | `GOLDEN` — 10 labelled cases with `must_block` expectations |
| `eval_harness.py` | DeepEval metrics + `OfflineJudge` + `run_eval()` scorecard + gate |
| `trace_logger.py` | ANSI trace helpers (honors `NO_COLOR`) |
| `demo.py` | CLI entry point for the five demos |

## Four PROD PATTERNS

- **Input Guardrail Chain** — composable checks; the first BLOCK short-circuits.
- **Output Validation Gate** — validate model output before it reaches the user.
- **Fail-Closed Default** — a guardrail that trips *or errors* BLOCKs; never pass on failure.
- **Eval Harness / Evaluation Frameworks** — golden dataset + metrics + pass/fail gate, run like tests.

---

# Eval-Driven Development (Half 2)

Once you HAVE an eval harness (Half 1), **eval-driven development (EDD)** is the
discipline of using it as your development LOOP — like Test-Driven Development,
but for AI answer quality. Add the failing eval case first; the fix is "done"
when the suite goes green.

## Run all EDD demos offline

```bash
PYTHONPYCACHEPREFIX=/tmp/s40_pycache DEEPEVAL_TELEMETRY_OPT_OUT=YES FAKE_LLM=1 python edd_demo.py all
```

Run a single demo: `FAKE_LLM=1 python edd_demo.py 3` (1..4 or `all`).
Run the CI gate directly: `FAKE_LLM=1 python ci_gate.py V2_FIXED` (exit 0) or
`FAKE_LLM=1 python ci_gate.py V1_BUGGY` (exit 1).

## The four EDD demos

1. **Eval-First (red → green)** — run the suite on `V1_BUGGY`: one case FAILS, the
   gate is red. Switch to `V2_FIXED`: the gate goes green. We wrote the failing
   case first; the fix is what makes it pass.
2. **Regression Dataset** — take the bug from demo 1, `add_from_failure()` it into a
   `RegressionDataset`, save + reload it, rerun `V2_FIXED`: the once-broken case is
   now a permanent, covered case that still passes.
3. **A/B Experiment** — `run_experiment` runs the SAME dataset against
   `{prompt_v1: V1_BUGGY, prompt_v2: V2_FIXED}`, diffs them case-by-case, and names
   the winner by gate then pass-rate.
4. **Eval Gate in CI** — `ci_gate` runs the suite on a variant and exits non-zero on a
   regression. `V2_FIXED` → exit 0 (merge allowed); `V1_BUGGY` → exit 1 (merge blocked).

## The deterministic offline mechanism

`agent_variants.py` defines two named variants over the SAME verified `SafeAgent`.
`V1_BUGGY` models a prompt regression: on exactly ONE question ("How long do
transfers take to settle?") it injects a `[HALLUCINATION]` marker, so the verified
grounding gate BLOCKs an ungrounded answer. Because that case is `must_block=False`,
the eval coverage assertion FAILS for V1. `V2_FIXED` injects nothing → the same case
is answered cleanly and PASSES. Same dataset, same harness, two deterministic scores
— that is the A/B. The verified `llm_client.py` is never modified.

## New EDD files

| File | What it is |
|---|---|
| `agent_variants.py` | `V1_BUGGY` / `V2_FIXED` variants over `SafeAgent` (deterministic bug injection) |
| `regression_cases.json` | Two "previously escaped bug" cases the regression dataset seeds from |
| `regression_dataset.py` | `RegressionDataset` — `seed`/`load`/`save`/`add_from_failure`, the growing suite |
| `eval_runner.py` | `run_suite()` → `RunReport` (timestamp, rows, pass count, gate); `save_run`/`load_run` |
| `compare_runs.py` | `compare()` → `ComparisonReport` (per-case IMPROVED/REGRESSED/SAME + winner) |
| `experiment.py` | `run_experiment()` — A/B across named variants, pick the winner |
| `ci_gate.py` | runnable CI gate; `sys.exit(0/1)` on the suite's gate verdict |
| `edd_demo.py` | CLI entry point for the four EDD demos |

## Four EDD PROD PATTERNS

- **Eval-First Development** — add the failing eval case BEFORE the fix; green = done.
- **Regression Dataset** — every escaped bug becomes a permanent case; it can't silently return.
- **Experiment Tracking / A-B Comparison** — score two variants on one dataset; pick by score (callbacks: S35 Best-of-N, S33 quality gate).
- **Eval Gate in CI** — a red suite exits non-zero and blocks the merge before a bad change ships.
