# plan.md — Review Radar

> The HOW. This file translates [spec.md](spec.md) into an architecture, a data
> contract, and a test plan. Every decision cites the acceptance criterion (AC#) it
> serves. If this plan and the spec disagree, the spec wins — fix the spec first.
> No application code lives here; code comes in `tasks.md` → implementation.

References: AC1–AC13 are the numbered acceptance criteria in [spec.md](spec.md) §5.

---

## 1. Architecture (exactly three files)

Per CLAUDE.md, the whole app is three Python files plus a couple of supporting
files. Each file has one job and imports all settings from `config.py`.

| File          | Responsibility | Serves |
|---------------|----------------|--------|
| `config.py`   | The single source of all settings. Calls `load_dotenv()` first (loads a local `.env` into the environment if present; no-op in production), then reads `ANTHROPIC_API_KEY`, `MODEL`, and `FAKE_LLM` from `os.environ`. Exposes the canned `FAKE_LLM` payload and tunable constants (theme count bounds, large-batch threshold). No business logic. | Convention 1 & 2; AC4, AC10 |
| `analyzer.py` | One pure-ish function `analyze_reviews(reviews: list[str]) -> dict`. Splits/cleans input, builds the prompt, calls Claude (or returns the fake payload), parses and validates the JSON, and enforces the fail-soft rules. Returns the data contract dict — never raises to the caller. | AC1–AC12 |
| `main.py`     | FastAPI app. `GET /` (HTML form + renders results), `GET /health` (liveness), `POST /analyze` (takes pasted text, calls `analyze_reviews`, renders or returns the result). No analysis logic of its own. | AC1, AC8, AC13 |

Supporting (not "application logic", allowed by stack/conventions):
- `requirements.txt` — pinned dependencies (includes `python-dotenv`).
- `.env.example` — documents `ANTHROPIC_API_KEY`, `MODEL`, `FAKE_LLM`.
- `.gitignore` — ignores the real `.env` so the secret is never committed (Convention 2).
- `tests/` — the test suite from §7.

**Data flow:** `main.py` receives raw pasted text → passes the raw string to
`analyzer.py` → `analyzer.py` owns line-splitting (AC3), analysis, validation, and
fail-soft → returns the contract dict → `main.py` renders it. Keeping the splitting
inside `analyzer.py` means the counting rules (AC3) are tested in one place.

---

## 2. The JSON data contract (what the model MUST return, and what `analyze_reviews` returns)

The model is asked to return **only** a JSON object in this exact shape. The same
shape is what `analyze_reviews` returns to `main.py`, after validation — so the rest
of the app has a stable contract (AC12: consistent shape on repeat runs).

```json
{
  "ok": true,
  "analyzed_count": 12,
  "sentiment": { "positive": 7, "negative": 3, "neutral": 2 },
  "themes": [
    { "label": "Slow startup time", "count": 5 },
    { "label": "Love the new dashboard", "count": 4 },
    { "label": "Login bug after update", "count": 3 }
  ],
  "summary": "Most users are happy with the new dashboard, but startup speed and a post-update login bug are dragging the release down. Prioritize the login fix.",
  "message": null
}
```

Field-by-field, with the criterion each field exists to satisfy:

| Field | Type | Rule | Serves |
|-------|------|------|--------|
| `ok` | bool | `true` for a normal result, `false` for a fail-soft/empty result. The UI branches on this. | AC8, AC11 |
| `analyzed_count` | int ≥ 0 | Number of reviews actually analyzed = number of non-empty lines. | AC2, AC3 |
| `sentiment` | object | Exactly the keys `positive`, `negative`, `neutral`, each int ≥ 0. **Must sum to `analyzed_count`.** | AC2 |
| `themes` | array | 0–5 items. Each item: `{ "label": str, "count": int }`. Target 3–5 when input allows; fewer is acceptable for tiny batches; **never more than 5**. | AC4, AC5 |
| `themes[].count` | int | 1 ≤ count ≤ `analyzed_count`. | AC5, AC6 |
| `summary` | str | A few sentences, plain English, action-oriented. | AC7 |
| `message` | str or null | Human-readable note. `null` on success; populated on empty/fail-soft (e.g. "Nothing to analyze", "That batch was too large to process in full"). | AC8, AC10, AC11 |

This is the **only** contract. `main.py` and the tests depend on these keys and on
the two invariants (sentiment sums to count; theme counts ≤ count).

---

## 3. Prompt strategy

Goal: get well-formed JSON matching §2 on the first try, English output, grounded in
the batch.

1. **System prompt** establishes the role ("You analyze batches of customer product
   reviews for a product manager") and hard rules:
   - Return **only** a single JSON object in the exact schema from §2 — no prose, no
     markdown fences. (Makes parsing reliable → AC11.)
   - `sentiment` counts must sum to the number of reviews given. (AC2)
   - Return **3 to 5 themes**, each with a `count` of how many reviews relate to it;
     never more than 5; a `count` never exceeds the number of reviews. (AC4–AC6)
   - `summary` = 2–4 sentences, plain English, says what the team should act on. (AC7)
   - Output in **English** even if some reviews are in another language. (spec §6)
2. **User message** carries the reviews as a numbered list, one per line, so the
   model can reason about "how many reviews" and tie counts to frequency (AC5). The
   exact count of reviews is stated explicitly in the message so the model anchors
   `analyzed_count`/sentiment totals to it (AC2).
3. **Determinism of shape, not wording:** request `temperature` low (e.g. 0–0.2) so
   the *shape* and counts are stable across runs even though wording may vary (AC12).
4. **Model:** `MODEL` from `config.py` (Claude Haiku per CLAUDE.md), so the choice is
   configurable and never hardcoded (Convention 1).
5. **Token budget:** request a max output sufficient for the small JSON object. The
   *input* is the variable part — see large-batch handling in §4/§2 (`message`, AC10).

> The prompt asks the model to do the counting/labeling. The code does **not**
> re-derive sentiment from the text; it only *validates* what came back (§4). This
> keeps `analyzer.py` small (CLAUDE.md) while the invariants still get enforced.

---

## 4. Fail-soft rules (AC8, AC10, AC11, AC12)

`analyze_reviews` must **never raise** to `main.py`. Every failure path returns a
valid §2 dict with `ok: false` and a friendly `message`. Order of checks:

1. **Clean input first (AC3).** Split the raw text on newlines, strip each line, drop
   empty/whitespace-only lines. The surviving list is the reviews; its length is
   `analyzed_count`.
2. **Empty batch (AC8).** If zero reviews survive → return immediately, no model call:
   `{ ok:false, analyzed_count:0, sentiment:{0,0,0}, themes:[], summary:"",
   message:"There were no reviews to analyze. Paste one review per line." }`
3. **Large batch (AC10).** If `analyzed_count` exceeds `LARGE_BATCH_LIMIT` (a constant
   in `config.py`), do **not** silently truncate. Return `ok:false` with
   `message:"That batch is too large to process in full (N reviews). Try a smaller
   batch."` — an explicit refusal, not a partial result dressed as complete.
4. **Model/transport failure (AC11).** Wrap the API call in try/except. On any
   exception (network, auth, timeout) → `ok:false`, generic friendly `message`, no raw
   stack trace or provider error surfaced.
5. **Unparseable / malformed JSON (AC11).** If the response isn't valid JSON, or is
   missing required keys, or has wrong types → `ok:false` with a "couldn't read the
   analysis" message. Never crash on `json.loads`.
6. **Invariant repair/validation (AC2, AC4, AC6).** On a parsed result:
   - If `sentiment` doesn't sum to `analyzed_count`, or keys are missing → treat as
     malformed (fail soft per step 5) rather than emitting a contract-violating dict.
   - Trim `themes` to at most 5 (AC4). Drop any theme whose `count` > `analyzed_count`
     or whose `count` < 1, or that is missing `label`/`count` (AC5, AC6).
   - These guarantees mean a *successful* (`ok:true`) result always satisfies the
     §2 invariants — which is exactly what AC12 (consistent shape) tests rely on.

`main.py` rule: it renders `message` when present and otherwise renders the three
outputs; it returns HTTP 200 with a friendly result even on fail-soft, so the user
never sees a 500 or raw error (AC11).

---

## 5. FAKE_LLM mode (Convention 5; supports offline testing of AC1–AC12)

`config.py` reads `FAKE_LLM` from the environment. When `FAKE_LLM=1`:

- `analyzer.py` skips the Anthropic call entirely and produces a result **derived
  from the actual cleaned input**, so the canned path still honors the contract:
  - `analyzed_count` = real count of non-empty lines (AC3).
  - `sentiment` = a deterministic split that **sums to `analyzed_count`** (AC2) —
    e.g. distribute the count across the three buckets by a fixed rule.
  - `themes` = a fixed 3-item list with `count` values clamped to ≤ `analyzed_count`
    (AC4–AC6).
  - `summary` = a fixed plain-English sentence (AC7).
- The empty-batch (AC8) and large-batch (AC10) checks run **before** the fake payload,
  so those paths behave identically with or without a key.
- Purpose: the app runs and the full test suite passes with **no API key and no
  network** (Convention 5), and tests are deterministic (AC12) because the fake output
  is a pure function of the input.

`FAKE_LLM` unset/`0` → real Anthropic path using `ANTHROPIC_API_KEY` (Convention 2).

---

## 6. Configuration (Convention 1 & 2)

All read in `config.py`, exposed as module constants; every other file imports them.
`config.py` calls `load_dotenv()` before reading anything, so a local `.env` (if it
exists) populates `os.environ`; in production the host's injected env vars are used and
the `.env` step is a harmless no-op. The real `.env` is gitignored (Convention 2).

| Setting | Source | Default | Serves |
|---------|--------|---------|--------|
| `ANTHROPIC_API_KEY` | env | none (required only when `FAKE_LLM` off) | Convention 2 |
| `MODEL` | env | Claude Haiku id | Convention 1; §3 |
| `FAKE_LLM` | env | off | Convention 5; §5 |
| `MIN_THEMES` / `MAX_THEMES` | constant | 3 / 5 | AC4 |
| `LARGE_BATCH_LIMIT` | constant | e.g. 300 | AC10 |
| `FAKE_PAYLOAD` builder | constant/fn | — | §5 |

No secret is hardcoded; nothing is configured in two places.

---

## 7. Test plan (each test maps to an acceptance criterion)

Tests run with `FAKE_LLM=1` (deterministic, offline). Each test asserts against the
§2 contract. The mapping is one-to-(one-or-more); every AC has at least one test.

| # | Test | Asserts | AC |
|---|------|---------|----|
| T1 | Analyze a normal multi-line batch | Result has `sentiment`, `themes`, `summary` keys all populated | AC1 |
| T2 | Sentiment sums | `positive+negative+neutral == analyzed_count` | AC2 |
| T3 | Blank/whitespace lines ignored | Input with empty lines → `analyzed_count` counts only non-empty lines; totals unaffected | AC3 |
| T4 | Theme count bound | `len(themes) <= 5` on a large/varied batch | AC4 |
| T5 | Theme shape | Every theme has a non-empty `label` (str) and a `count` (int) | AC5 |
| T6 | Theme count never exceeds total | Every `themes[i].count <= analyzed_count` and `>= 1` | AC6 |
| T7 | Summary quality | `summary` is a non-empty string, within a sane length bound (a few sentences) | AC7 |
| T8 | Empty submission | `""` or only-whitespace input → `ok:false`, `analyzed_count==0`, friendly `message`, no exception | AC8 |
| T9 | Single review | One-line input → `ok:true`, all three outputs present, no exception | AC9 |
| T10 | Large batch is explicit | Input above `LARGE_BATCH_LIMIT` → `ok:false` with an explicit "too large" `message`, not a silent partial result | AC10 |
| T11 | Malformed model output fails soft | Force the parser onto a non-JSON / wrong-shape string → returns `ok:false` dict, never raises | AC11 |
| T12 | Bad-key / wrong-type payload fails soft | Missing keys or wrong types → `ok:false`, no crash | AC11 |
| T13 | Shape consistency on repeat | Same input twice → identical set of contract keys and same invariants hold both times | AC12 |
| T14 | Health endpoint | `GET /health` returns success without any review input | AC13 |
| T15 | `POST /analyze` end-to-end | Posting form text returns a 200 with rendered/JSON result (never a 500), exercising the `main.py` ↔ `analyzer.py` seam | AC1, AC11 |

Coverage check: AC1✓(T1,T15) AC2✓(T2) AC3✓(T3) AC4✓(T4) AC5✓(T5) AC6✓(T6)
AC7✓(T7) AC8✓(T8) AC9✓(T9) AC10✓(T10) AC11✓(T11,T12,T15) AC12✓(T13) AC13✓(T14).
**All 13 acceptance criteria are covered.**

---

## 8. Decisions deliberately deferred to `tasks.md` / implementation
- Exact HTML/CSS of the page (spec forbids a frontend framework; styling is not a
  spec concern).
- Exact wording of prompts and friendly messages.
- The precise deterministic split rule used by `FAKE_PAYLOAD`.
- Dependency pin versions in `requirements.txt`.

These are implementation details and carry no acceptance criterion; fixing them here
would be over-specifying.
