# Session 45 — Domain-Driven Design (reference artifacts)

**Model the world before you build it.** This folder is the worked example for
the session: a messy stakeholder brief turned into a clean, shared model using AI
as the modelling partner — then code that speaks the business's own words.

## What's here

| File | What it is |
|------|------------|
| `CLAUDE.md` | The agent briefing read before every prompt — points the AI at the glossary + context map and enforces the language rules. |
| `messy_brief.md` | The raw input: 4 stakeholders, 4 vocabularies, the same word meaning 4 things. |
| `prompts/01_extract_ubiquitous_language.md` | Prompt 1 — AI surfaces the ubiquitous language from the brief. |
| `prompts/02_find_bounded_contexts.md` | Prompt 2 — AI groups terms into bounded contexts + a collision table. |
| `prompts/03_draw_context_map.md` | Prompt 3 — AI draws how contexts hand off (and translate) work. |
| `prompts/04_generate_domain_model.md` | Prompt 4 — AI generates the Ordering model in the glossary's words. |
| `GLOSSARY.md` | **The PROD-PATTERN artifact** — the ubiquitous language. Feed this to every coding session. |
| `context_map.md` | The four bounded contexts + the "Order" collision + the context map. |
| `domain_model.py` | The language-rich Ordering model (Prompt 4's output). Entities, value objects, invariants. |
| `anemic_antipattern.py` | What the AI writes WITHOUT a glossary: `UserManager`, `process_data`, a free-text `status`. The contrast. |
| `demo.py` | Runs the model and watches the two invariants fire. No key, no network. |

## Run it (no API key needed)

```bash
python demo.py
```

You should see: an Order built in MealHop's own words, two invariants blocking
illegal moves (confirm-when-empty, place-before-confirm), and the anemic model
happily setting `status = "placed"` with no rule to stop it.

## The production pattern

**Ubiquitous Language Glossary as an AI context artifact.** `GLOSSARY.md` +
`context_map.md` are checked into the repo and referenced from `CLAUDE.md`, so
every prompt the AI sees inherits the business's vocabulary. When the AI invents
`UserManager`, you don't argue — you point it back at the glossary. This is the
S44 spec-driven discipline extended: the spec is now written in the domain's
language, and that language is enforced on every generated file.

## How the prompts map to the loop

```
messy_brief.md
   │  Prompt 1  →  GLOSSARY.md         (the words)
   │  Prompt 2  →  bounded contexts    (where each word has ONE meaning)
   │  Prompt 3  →  context_map.md      (how contexts translate at the boundary)
   └  Prompt 4  →  domain_model.py     (the words become code)
```
