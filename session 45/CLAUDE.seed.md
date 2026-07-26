# CLAUDE.md — MealHop  (LIVE-BUILD SEED — the only file in the folder at the start)

## Project
MealHop is a meal-subscription service. Before we write any feature code, we model
the domain with Domain-Driven Design. Model first, code last.

## How we work (read before every prompt)
- We build artifacts IN ORDER: brief → glossary → contexts → context map → model.
- One step per prompt. Create ONLY the file I ask for. No code until the model step.
- Use the BUSINESS's words. Banned generic names: User, Manager, Service, Handler,
  Data, Process, process_data, update, free-text `status` strings.
- One word, one meaning PER context. The same word may mean different things in
  different contexts — translate at the boundary, never merge contexts to "simplify".
- Mark every type ENTITY (has identity) or VALUE OBJECT (interchangeable, immutable).
- Money is never a float. Don't add a field I didn't ask for — ask first, then we add
  it to the glossary BEFORE the code.

## The build plan (what we will create, in this order)
1. `messy_brief.md`  — I paste the raw stakeholder brief.
2. `GLOSSARY.md`     — extract the ubiquitous language from the brief. (Prompt 1)
3. `context_map.md`  — bounded contexts + a collision table + the context map. (Prompts 2–3)
4. `domain_model.py` — the Ordering context, in the glossary's words, with invariants. (Prompt 4)
5. `demo.py`         — prove the invariants fire.

## After the model exists
Harden THIS file: add `GLOSSARY.md` and `context_map.md` as the source of truth for
language, so every later prompt inherits the business's words automatically.
