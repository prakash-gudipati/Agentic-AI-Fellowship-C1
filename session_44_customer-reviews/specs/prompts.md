# prompts.md — Review Radar

> Copy-paste implementation prompts, one per task in [tasks.md](tasks.md). Run them
> **in order, one at a time**. Each prompt is self-contained, names the task and the
> acceptance criteria (AC#) it satisfies, and ends with a **VERIFY** gate. Do not start
> the next prompt until the current VERIFY passes. If a VERIFY fails, fix within the
> same task before moving on.
>
> Ground rules to paste-or-assume for every prompt: *Follow CLAUDE.md and
> [plan.md](plan.md). Spec-driven: the spec wins. Touch only the file(s) named in the
> task. Type hints, small functions, comments explain WHY. Do not implement future
> tasks.*

---

## How to use this file
1. Copy the **Prompt** block for the current task into Claude Code.
2. When it finishes, run the **VERIFY** block yourself (or ask Claude to run it).
3. If VERIFY passes, tick the task box in [tasks.md](tasks.md) and move to the next.
4. If it fails, paste the failure output back and ask for a fix — still within this task.

PowerShell note: set env vars per-command like `$env:FAKE_LLM=1; pytest`.

---

## T0 — Scaffold the project
**Prompt:**
> Implement task **T0** from [tasks.md](tasks.md). Create the project skeleton only —
> no business logic yet. Files: `requirements.txt` (FastAPI, an ASGI server like
> uvicorn, the Anthropic SDK, `python-dotenv`, `pytest`, `httpx` for endpoint tests —
> all pinned), `.env.example` documenting `ANTHROPIC_API_KEY`, `MODEL`, `FAKE_LLM`,
> `.gitignore` (ignore `.env`, `__pycache__/`, venv), and empty `config.py`,
> `analyzer.py`, `main.py`, plus `tests/__init__.py`. No secrets hardcoded.
> Serves Conventions 1 & 2.

**VERIFY:**
- [ ] All files exist: `requirements.txt`, `.env.example`, `.gitignore`, `config.py`,
  `analyzer.py`, `main.py`, `tests/__init__.py`.
- [ ] In a fresh venv: `pip install -r requirements.txt` succeeds.
- [ ] `.gitignore` contains `.env`.

---

## T1 — config.py: settings + dotenv
**Prompt:**
> Implement task **T1**. In `config.py` only: call `load_dotenv()`, then read
> `ANTHROPIC_API_KEY`, `MODEL` (default to the Claude Haiku model id), and `FAKE_LLM`
> (boolean-ish) from `os.environ`. Define constants `MIN_THEMES=3`, `MAX_THEMES=5`,
> `LARGE_BATCH_LIMIT=300`. No business logic, no hardcoded secret. Serves Conventions
> 1 & 2, AC4, AC10.

**VERIFY:**
- [ ] `python -c "import config; print(config.MODEL, config.MIN_THEMES, config.MAX_THEMES, config.LARGE_BATCH_LIMIT)"`
  runs with no env set and prints sensible defaults; `FAKE_LLM` resolves to off.
- [ ] Putting `MODEL=test-model` in a local `.env` makes `config.MODEL == "test-model"`.

---

## T2 — analyzer.py: input cleaning + counting
**Prompt:**
> Implement task **T2**. In `analyzer.py` only: add a helper that takes raw pasted text,
> splits on newlines, strips each line, drops empty/whitespace-only lines, and returns
> the list of reviews. Add the `analyze_reviews(reviews_or_text)` skeleton that uses it
> to compute `analyzed_count`, returning a stub dict matching the plan §2 contract keys
> for now. Serves AC3.

**VERIFY:**
- [ ] Quick check: input `"good\n\n   \nbad\n"` → cleaned list `["good", "bad"]`,
  `analyzed_count == 2`.
- [ ] Returned dict has the keys: `ok`, `analyzed_count`, `sentiment`, `themes`,
  `summary`, `message`.

---

## T3 — analyzer.py: empty + large-batch fail-soft
**Prompt:**
> Implement task **T3**. In `analyzer.py` only, before any model call: if zero reviews
> survive cleaning, return `ok:false`, `analyzed_count:0`, `sentiment` all zeros,
> `themes:[]`, empty `summary`, and a friendly "nothing to analyze" `message`. If the
> count exceeds `config.LARGE_BATCH_LIMIT`, return `ok:false` with an explicit
> "too large" `message` and no silent truncation. Never raise. Serves AC8, AC10.

**VERIFY:**
- [ ] `""` and `"   \n\t"` → `ok==False`, `analyzed_count==0`, `message` set, no exception.
- [ ] A batch of `LARGE_BATCH_LIMIT + 1` lines → `ok==False`, `message` mentions too large.
- [ ] A normal small batch still passes through (does not hit either branch).

---

## T4 — config.py + analyzer.py: FAKE_LLM payload
**Prompt:**
> Implement task **T4**. When `config.FAKE_LLM` is on, `analyze_reviews` must skip the
> Anthropic call and build a result **derived from the cleaned input**: `analyzed_count`
> = real count; a deterministic `sentiment` split that **sums to `analyzed_count`**; a
> fixed list of 3 themes each with a `count` clamped to ≤ `analyzed_count`; a fixed
> plain-English `summary`; `message:null`; `ok:true`. Put the payload builder in
> `config.py`, wire it in `analyzer.py`. The empty/large-batch checks from T3 still run
> first. Serves Convention 5, AC2, AC4–AC7.

**VERIFY:**
- [ ] With `FAKE_LLM=1`, a 5-line batch → `ok==True`; `sentiment` sums to 5; `len(themes) <= 5`;
  every `theme.count <= 5`; `summary` non-empty.
- [ ] Same input twice → identical result (deterministic).

---

## T5 — analyzer.py: real Claude call + prompt
**Prompt:**
> Implement task **T5**. In `analyzer.py` only, the `FAKE_LLM`-off path: build the
> system + user prompt per [plan.md](plan.md) §3 — JSON-only output in the §2 schema,
> reviews as a numbered list with the review count stated explicitly, English output,
> low temperature — and call the Anthropic SDK using `config.MODEL` and
> `config.ANTHROPIC_API_KEY`. Return the raw model text to be parsed in T6 (do not parse
> yet). Serves AC1, AC2, AC5, AC7.

**VERIFY:**
- [ ] Code review: prompt enforces JSON-only schema, numbered reviews, stated count,
  English, low temperature; model id comes from `config`, not hardcoded.
- [ ] (Optional, needs a real key) `FAKE_LLM` off on a 3-line batch returns text that is
  valid JSON in the §2 shape.

---

## T6 — analyzer.py: parse, validate, repair (fail-soft)
**Prompt:**
> Implement task **T6**. In `analyzer.py` only: parse the model output as JSON inside
> try/except. On any exception, non-JSON, missing keys, or wrong types → return
> `ok:false` with a friendly `message`; never raise. On a parsed result: reject (fail
> soft) if `sentiment` doesn't sum to `analyzed_count`; trim `themes` to at most
> `config.MAX_THEMES`; drop any theme missing `label` or with `count` < 1 or
> `count` > `analyzed_count`. Guarantee that any `ok:true` result satisfies the §2
> invariants. Serves AC2, AC4, AC6, AC11, AC12.

**VERIFY:**
- [ ] Feed a non-JSON string and a wrong-shape/missing-keys dict to the parser →
  `ok==False`, no crash.
- [ ] Feed a valid payload with 7 themes and one `count` > count → result has ≤5 themes
  and no theme `count` exceeds `analyzed_count`.
- [ ] A payload whose `sentiment` doesn't sum to count → `ok==False`.

---

## T7 — main.py: app, GET / and GET /health
**Prompt:**
> Implement task **T7**. In `main.py` only: create the FastAPI app. `GET /` returns the
> HTML form (plain HTML with inline CSS/JS — no frontend framework) with a textarea for
> pasted reviews and a submit. `GET /health` returns a simple liveness response that
> needs no input. No analysis logic here. Serves AC13, AC1.

**VERIFY:**
- [ ] Run the app; `GET /health` returns success (200) with no body required.
- [ ] `GET /` returns HTML containing the review textarea form.

---

## T8 — main.py: POST /analyze wiring
**Prompt:**
> Implement task **T8**. In `main.py` only: add `POST /analyze` that reads the pasted
> textarea content, calls `analyze_reviews`, and renders the result — sentiment split,
> themes with counts, summary — or renders the `message` when `ok:false`. Always return
> HTTP 200; never surface a 500 or a raw error. Serves AC1, AC8, AC10, AC11.

**VERIFY:**
- [ ] Posting a normal batch returns 200 with sentiment, themes (with counts), summary.
- [ ] Posting empty input returns 200 showing the friendly "nothing to analyze" message.
- [ ] No request produces a 500 or stack trace in the response.

---

## T9 — tests/: full suite mapping to acceptance criteria
**Prompt:**
> Implement task **T9**. Create `tests/test_review_radar.py` implementing tests T1–T15
> from [plan.md](plan.md) §7, run with `FAKE_LLM=1` (offline, no key). One or more
> assertions per acceptance criterion, asserting against the §2 contract. Include the
> `GET /health` and `POST /analyze` endpoint tests using a test client. Serves AC1–AC13.

**VERIFY:**
- [ ] `$env:FAKE_LLM=1; pytest -q` → all tests pass with no network/key.
- [ ] Each of plan §7's T1–T15 is present; coverage table maps every AC to a passing test.

---

## T10 — Verify: end-to-end run
**Prompt:**
> Implement task **T10** (no new files). Help me run the app end-to-end with
> `FAKE_LLM=1` and walk the three flows below. Then summarize pass/fail against AC1–AC13
> and confirm all task boxes in [tasks.md](tasks.md) are checked.

**VERIFY (manual, in a browser):**
- [ ] Start app with `FAKE_LLM=1`. Paste a sample batch → see sentiment split, 3–5 themes
  with counts, and a summary.
- [ ] Submit empty input → friendly "nothing to analyze" message, no crash.
- [ ] `GET /health` → OK.
- [ ] `$env:FAKE_LLM=1; pytest -q` is green.
- [ ] (Optional) Set a real `ANTHROPIC_API_KEY`, `FAKE_LLM` off, repeat the paste flow →
  live result in the §2 shape, no 500.
- [ ] All T0–T9 boxes checked in [tasks.md](tasks.md).

---

### Sequencing rule (the whole point of this file)
T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9 → T10. **One task per prompt. Pass its
VERIFY before the next.** A failing VERIFY is fixed inside the same task — never deferred.
