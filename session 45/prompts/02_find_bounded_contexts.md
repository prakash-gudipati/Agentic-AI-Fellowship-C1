# PROMPT 2 — Find the Bounded Contexts

You are a domain modelling assistant. Do NOT write any code.

Using the glossary we just built, group the terms into BOUNDED CONTEXTS — areas of
the business where each term has ONE precise, consistent meaning.

For each bounded context produce:
- Context name (e.g. Ordering, Kitchen, Delivery, Billing)
- One sentence: what this context is responsible for
- The terms that live inside it, with the meaning they have HERE
- The terms it deliberately does NOT care about

Then build a COLLISION TABLE for any term that appears in more than one context:
| Term | In <Context A> it means... | In <Context B> it means... | ... |

The point: the same word is allowed to mean different things in different contexts,
as long as the boundary is explicit. Make every boundary explicit.
