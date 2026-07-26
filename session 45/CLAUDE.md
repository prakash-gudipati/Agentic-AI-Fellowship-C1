# CLAUDE.md — MealHop (reference project for Session 45)
# The briefing Claude Code reads BEFORE every prompt.
# Everything below is derived from the artifacts in this repo — not re-invented per prompt.

## What this project is
MealHop is a meal-subscription service. Subscribers pick a MealPlan, assemble an
Order of Meals for a DeliveryWindow, and the Order is fulfilled by the Kitchen,
Delivery, and Billing parts of the business. This repo is the worked example for
Domain-Driven Design: we model the domain FIRST, then let the model drive the code.

## Source-of-truth files (read these first, every session)
| File | What it governs | Rule |
|------|-----------------|------|
| `GLOSSARY.md` | the LANGUAGE — every name in the code | If a word isn't here, don't invent it — ask, then add it here FIRST. |
| `context_map.md` | the BOUNDARIES — where each word has one meaning | A term may mean different things in different contexts. Never merge contexts to "simplify". |
| `messy_brief.md` | the raw domain input | Where the language was extracted from. Reference, not law. |
| `domain_model.py` | the Ordering context in code | The shape every new code file must match. |

## Spec-driven discipline (carried over from S44)
- The spec is the source of truth for BEHAVIOUR. The glossary is the source of truth for LANGUAGE.
- When code and spec disagree, fix the spec first. When code and glossary disagree, fix the glossary first.
- Implement one task at a time. Read the diff. Run it. Then continue.

## Domain rules Claude MUST follow (derived from GLOSSARY.md + context_map.md)
1. **Use the ubiquitous language.** Class, method, and field names come from `GLOSSARY.md`.
   Banned generic names: `User`, `Manager`, `Service`, `Handler`, `Data`, `Process`,
   `process_data`, `update`, free-text `status` strings.
2. **One word, one meaning, per context.** "Order" means a basket in Ordering, a
   PrepTicket in Kitchen, a Stop in Delivery, a Charge in Billing. Do not build one
   Order class that serves all four — translate at the boundary (see `context_map.md`).
3. **Mark every type.** ENTITY (has identity, e.g. Subscriber, Order) or VALUE OBJECT
   (interchangeable, immutable, e.g. Money, DeliveryWindow, MealPlan) in a one-line docstring.
4. **Name methods for business actions.** `confirm`, `place`, `add_meal` — not
   `process`, `update`, `manage`.
5. **Enforce invariants in the model**, not in calling code. Current invariants:
   I1 a confirmed Order has ≥1 Meal; I2 an Order can't be placed before it's confirmed.
6. **Money is never a float.** Use the `Money` value object (minor units / paise).
7. **Don't add fields the glossary doesn't mention.** If you think one is missing,
   stop and ask — we update `GLOSSARY.md` before the code.

## How to extend this project
- New concept? → add it to `GLOSSARY.md` first, place it in a context in `context_map.md`,
  THEN write the code in those words.
- New context (Kitchen / Delivery / Billing)? → model it separately and define the
  translation from a confirmed Order at the boundary.

## Run
```bash
python demo.py    # builds an Order in the business's words; both invariants fire. No key, no network.
```
