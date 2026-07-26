# PROMPT 1 — Extract the Ubiquitous Language

You are a domain modelling assistant. Do NOT write any code.

Read the attached stakeholder brief (messy_brief.md). Your job is to surface the
UBIQUITOUS LANGUAGE — the small set of nouns and verbs the business actually uses
for the things that matter.

Produce a draft GLOSSARY in a table with these columns:
| Term | Plain-English meaning | Said by whom | Notes / conflicts |

Rules:
- Use the business's own words. Do not invent generic tech words
  (no "User", "Manager", "Handler", "Service", "Data", "Process").
- When two words seem to mean the same thing, list them on one row and flag it.
- When ONE word seems to mean different things to different people, flag it loudly —
  we will resolve that with bounded contexts in the next step.
- Ask me up to 3 clarifying questions about any term you cannot pin down.

End with a one-line summary: which single term is the most overloaded?
