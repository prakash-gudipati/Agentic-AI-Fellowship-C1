# spec.md — Review Radar

> Source of truth. This file describes WHAT and WHY only — no technology choices,
> no implementation, no UI design. If the code disagrees with this spec, the spec wins.

## 1. Problem statement
After every product release, a flood of customer reviews arrives — app store
ratings, support replies, survey free-text, social comments. A product manager is
expected to know, quickly, "what did customers actually think this time?" But the
volume makes that impossible to do by hand: the signal (the recurring complaints,
the thing everyone loved, the one bug that keeps coming up) is buried under
hundreds of individual opinions. Reading them one by one is slow, inconsistent,
and easy to bias toward whatever was read last. The PM ends up either guessing or
spending hours they don't have, and the team's next priorities are set on a hunch.

## 2. Who it is for
The primary user is a **product manager** on a software product team who needs a
fast, honest read on customer sentiment right after a release — without writing
code, running a spreadsheet, or waiting on an analyst. Secondary readers are the
people the PM reports the result to: engineering leads, designers, and execs who
want the headline, not the raw reviews.

## 3. Goal (one sentence)
Turn a raw batch of customer reviews into an at-a-glance read — overall sentiment
split, the top recurring themes, and a short action-oriented summary — in a single
step.

## 4. User stories
1. **As a PM**, I want to paste a batch of reviews (one per line) and get back the
   overall sentiment split, so I know at a glance whether this release landed well.
2. **As a PM**, I want to see the top recurring themes and complaints with how many
   reviews mention each, so I can tell what's a widespread issue versus a one-off.
3. **As a PM**, I want a short plain-English summary I can paste into a release
   recap, so I can brief my team without rewriting anything.
4. **As a PM**, I want the tool to still give me a usable result when some lines are
   blank, junk, or oddly formatted, so messy real-world input doesn't break my run.
5. **As an engineering lead** reading the PM's recap, I want the themes tied to
   counts, so I can trust the priorities are driven by frequency, not by the loudest
   single review.

## 5. Acceptance criteria (numbered, testable)
Each criterion is phrased so it can pass or fail objectively.

1. Given a batch of one or more reviews (one per line) submitted for analysis, the
   system returns a result containing all three outputs: a sentiment split, a list
   of themes, and a summary.
2. The sentiment split reports three counts — **positive**, **negative**, and
   **neutral** — and those three counts sum to the number of reviews that were
   analyzed.
3. The system treats each non-empty line as exactly one review. Blank lines and
   lines containing only whitespace are ignored and are not counted in any total.
4. The themes output contains between **3 and 5** themes when there are enough
   reviews to support that many; with very few reviews it may return fewer, but
   never more than 5.
5. Each theme includes (a) a short human-readable label, and (b) a **count** of how
   many reviews in the batch relate to that theme.
6. No theme reports a count greater than the total number of analyzed reviews.
7. The summary is plain English, is action-oriented (states what the team should
   pay attention to), and is short enough to read at a glance (a few sentences, not
   a wall of text).
8. Given an empty submission (no reviews, or only blank/whitespace lines), the
   system does not error — it returns a clear, friendly message saying there was
   nothing to analyze.
9. Given a single review, the system still returns all three outputs without error.
10. Given a large batch (hundreds of reviews), the system returns a complete result
    without truncating the input silently; if the batch is too large to process in
    full, the system says so explicitly rather than returning a partial result as if
    it were complete.
11. If the underlying analysis cannot produce a well-formed result for any reason,
    the system fails soft: it returns a clear message to the user and never crashes
    or shows a raw internal error.
12. The same batch submitted twice produces results that are consistent in shape —
    the same three outputs, the same counting rules — even if the wording of the
    summary or theme labels varies.
13. There is a way to confirm the service is alive and ready (a basic health check)
    that responds without requiring any review input.

## 6. Out of scope (explicitly NOT in this product)
- **No accounts, login, or per-user history.** Every run is stateless.
- **No saving or database.** Results are not stored or retrievable later.
- **No file upload or integrations.** Input is pasted text only — no CSV import, no
   pulling reviews from app stores, support tools, or social APIs.
- **No per-review drill-down or editing.** The product reports on the batch as a
   whole, not a managed list of individual reviews.
- **No charts, dashboards, or exports.** The result is read on the page; no PDF,
   no email, no scheduled reports.
- **No multi-language guarantee.** Reviews are assumed to be in **English**;
   non-English input may still run but correct results are not promised.
- **No trend tracking over time.** The tool analyzes one batch in isolation; it does
   not compare this release to a previous one.

## 7. Open questions
- None blocking. Batch-size handling is specified as best-effort (criterion 10);
  if a future hard cap is desired, add it here and update the acceptance criteria
  before changing code.
