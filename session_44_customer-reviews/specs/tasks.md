# tasks.md — Review Radar

> The ordered build list. Derived from [plan.md](plan.md), which is derived from
> [spec.md](spec.md). Each task is the smallest sensible unit: it names the file(s) it
> touches, the acceptance criteria (AC#) or convention it satisfies, and how to verify
> it on its own. Do them in order — later tasks assume earlier ones. Check the box when
> a task is done **and** its verification passes.
>
> AC# refers to [spec.md](spec.md) §5. Conventions refer to CLAUDE.md.

---

## T0 — Scaffold the project
- [x] **Files:** `requirements.txt`, `.env.example`, `.gitignore`, empty `config.py`,
  `analyzer.py`, `main.py`, `tests/` dir with empty `__init__.py`.
- **Does:** Create the repo skeleton. `requirements.txt` lists FastAPI + server,
  Anthropic SDK, `python-dotenv`, and a test runner (pinned). `.env.example` documents
  `ANTHROPIC_API_KEY`, `MODEL`, `FAKE_LLM`. `.gitignore` ignores `.env`, `__pycache__`,
  venv.
- **Serves:** Conventions 1 & 2 (config separation, secrets via env); plan §1.
- **Verify:** `pip install -r requirements.txt` succeeds in a fresh venv; the six files
  exist; `git status` shows `.env` would be ignored.

## T1 — config.py: settings + dotenv
- [x] **File:** `config.py`
- **Does:** Call `load_dotenv()`, then read `ANTHROPIC_API_KEY`, `MODEL` (default Claude
  Haiku id), `FAKE_LLM` from `os.environ`. Define constants `MIN_THEMES=3`,
  `MAX_THEMES=5`, `LARGE_BATCH_LIMIT` (e.g. 300). No business logic, no secrets
  hardcoded.
- **Serves:** Conventions 1 & 2; AC4 (theme bounds), AC10 (large-batch limit); plan §6.
- **Verify:** `import config` works with no env set (uses defaults, `FAKE_LLM` off); a
  local `.env` value is picked up; `config.MODEL` and the numeric constants are present.

## T2 — analyzer.py: input cleaning + counting
- [x] **File:** `analyzer.py`
- **Does:** Helper that splits raw text on newlines, strips each line, drops
  empty/whitespace-only lines, returns the review list. `analyze_reviews` skeleton uses
  it to compute `analyzed_count`. Returns a stub contract dict for now.
- **Serves:** AC3 (blank lines ignored, one review per non-empty line); plan §1, §4 step 1.
- **Verify:** Unit-test the helper: mixed input with blank/whitespace lines yields only
  non-empty lines; `analyzed_count` equals that length. (Test T3.)

## T3 — analyzer.py: empty + large-batch fail-soft
- [x] **File:** `analyzer.py`
- **Does:** Before any model call — if zero reviews survive, return `ok:false`,
  `analyzed_count:0`, friendly "nothing to analyze" `message`. If count >
  `LARGE_BATCH_LIMIT`, return `ok:false` with an explicit "too large" `message` (no
  silent truncation).
- **Serves:** AC8 (empty), AC10 (large batch explicit); plan §4 steps 2–3.
- **Verify:** Empty/whitespace input → `ok:false`, count 0, message set, no exception
  (Test T8). Over-limit input → `ok:false` with "too large" message (Test T10).

## T4 — config.py + analyzer.py: FAKE_LLM payload
- [ ] **Files:** `config.py` (payload builder), `analyzer.py` (wire it in)
- **Does:** When `FAKE_LLM` is on, build a result **derived from the cleaned input**:
  `analyzed_count` = real count; `sentiment` split that **sums to count**; fixed 3 themes
  with `count` clamped ≤ count; fixed plain-English `summary`. Empty/large checks (T3)
  still run first.
- **Serves:** Convention 5; AC2, AC4–AC7 via deterministic offline output; plan §5.
- **Verify:** With `FAKE_LLM=1`, a normal batch returns a full contract dict; sentiment
  sums to count; ≤5 themes; counts ≤ count (Tests T1, T2, T4, T6, T7).

## T5 — analyzer.py: real Claude call + prompt
- [ ] **File:** `analyzer.py`
- **Does:** When `FAKE_LLM` is off, build the system + user prompt per plan §3 (JSON-only
  schema, reviews as numbered list, stated count, English, low temperature) and call the
  Anthropic SDK with `config.MODEL`. Return the raw model text to the parsing step (T6).
- **Serves:** AC1, AC2, AC5, AC7 (model produces the contract); plan §3.
- **Verify:** With a real key (manual/optional), a small batch returns parseable JSON;
  no key needed for CI since FAKE_LLM path covers the suite. Code reviewed for prompt
  rules.

## T6 — analyzer.py: parse, validate, repair (fail-soft)
- [ ] **File:** `analyzer.py`
- **Does:** Parse model output as JSON inside try/except. On exception (transport),
  non-JSON, missing keys, or wrong types → `ok:false` friendly `message`, never raise.
  On a parsed result: reject if `sentiment` doesn't sum to `analyzed_count`; trim
  `themes` to ≤5; drop themes with bad/missing `count` (<1 or >count) or missing
  `label`. A successful `ok:true` dict always satisfies the §2 invariants.
- **Serves:** AC2, AC4, AC6, AC11, AC12; plan §4 steps 4–6.
- **Verify:** Feed malformed / wrong-shape payloads → `ok:false`, no crash (Tests T11,
  T12). Valid payload → invariants hold (Tests T2, T4, T6, T13).

## T7 — main.py: app, GET / and GET /health
- [ ] **File:** `main.py`
- **Does:** Create the FastAPI app. `GET /` serves the HTML form (plain HTML, inline
  CSS/JS, no framework). `GET /health` returns a simple liveness response needing no
  input.
- **Serves:** AC13 (health check), AC1 (form entry point); plan §1.
- **Verify:** `GET /health` returns success with no body required (Test T14); `GET /`
  returns the form HTML.

## T8 — main.py: POST /analyze wiring
- [ ] **File:** `main.py`
- **Does:** `POST /analyze` reads the pasted text, calls `analyze_reviews`, and renders
  the result (or the `message` when `ok:false`). Always returns HTTP 200 — never a 500
  or raw error to the user.
- **Serves:** AC1 (all three outputs shown), AC8/AC10/AC11 (friendly messages, no 500);
  plan §1, §4.
- **Verify:** Posting form text returns 200 with rendered result; empty/over-limit/forced
  error inputs return 200 with the friendly message, never a stack trace (Test T15).

## T9 — tests/: full suite mapping to acceptance criteria
- [ ] **File:** `tests/test_review_radar.py` (run with `FAKE_LLM=1`)
- **Does:** Implement T1–T15 from plan §7 — one or more assertions per acceptance
  criterion, against the §2 contract. Includes the `POST /analyze` and `/health`
  endpoint tests.
- **Serves:** AC1–AC13 (coverage verification); plan §7.
- **Verify:** `pytest` passes with `FAKE_LLM=1`, no network/key. The coverage table in
  plan §7 maps every AC to at least one passing test.

## T10 — Verify: end-to-end run
- [ ] **Files:** none new (run the app)
- **Does:** Start the app locally with `FAKE_LLM=1`. In a browser: paste a sample batch
  → see sentiment split, 3–5 themes with counts, and a summary. Submit empty input → see
  the friendly "nothing to analyze" message. Hit `/health` → OK. Then (optional) repeat
  with a real `ANTHROPIC_API_KEY` and `FAKE_LLM` off to confirm the live path.
- **Serves:** AC1–AC13 end-to-end; confirms the spec's one-step goal is met.
- **Verify:** All three flows behave as above; no crash or 500 in any of them; full
  `pytest` suite green. Tasks T0–T9 boxes all checked.

---

### Acceptance-criteria coverage across tasks
AC1→T4,T7,T8,T10 · AC2→T4,T6 · AC3→T2 · AC4→T1,T6 · AC5→T5,T9 · AC6→T6 ·
AC7→T4,T5 · AC8→T3 · AC9→T9 · AC10→T3 · AC11→T6,T8 · AC12→T6,T9 · AC13→T7.
Every acceptance criterion is satisfied by at least one task; the suite in T9 and the
manual run in T10 verify the whole set.
