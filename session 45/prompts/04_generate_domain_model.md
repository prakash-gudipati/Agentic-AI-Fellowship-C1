# PROMPT 4 — Generate the Domain Model (ONE context)

You are a senior Python engineer who practises Domain-Driven Design.

Generate a small Python domain model for the **Ordering** bounded context ONLY,
using EXACTLY the names from our GLOSSARY.md. This is the moment the language
becomes code.

Requirements:
- Use the ubiquitous language for every class, method, and field.
  (Subscriber, MealPlan, DeliveryWindow, Money, Order, etc. — NOT User, Manager, Data.)
- Mark each type as an ENTITY (has identity) or a VALUE OBJECT (defined by its values,
  interchangeable, immutable) in a one-line docstring.
- The Order is the aggregate root. Give it domain methods named for business actions
  (e.g. add_meal, confirm, place) — not generic ones (no process, no update, no manage).
- Enforce at least one INVARIANT (a rule that must always be true), e.g.
  "an Order cannot be placed until it is confirmed" or
  "a confirmed Order must contain at least one meal".
- Pure standard library only. No database, no framework, no LLM call.
- A short docstring at the top naming the bounded context and listing the invariants.

Do not add fields the glossary does not mention. If you think one is missing,
ask me first — we update the GLOSSARY before the code.
