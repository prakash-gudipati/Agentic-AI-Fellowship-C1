# PROMPT 3 — Draw the Context Map

You are a domain modelling assistant. Do NOT write production code.

Draw a CONTEXT MAP showing how the bounded contexts hand work to each other.

For each handoff between two contexts, state:
- Direction (which context sends, which receives)
- What crosses the boundary (and in WHOSE language it is expressed)
- The TRANSLATION that happens at the boundary
  (e.g. Ordering's "confirmed Order" becomes Kitchen's "PrepTicket")

Output:
1. A simple ASCII diagram of the contexts and the arrows between them.
2. A table: | From | To | What crosses | Translated into |

Keep it to the contexts we actually found — do not invent new ones.
