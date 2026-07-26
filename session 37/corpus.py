"""
Session 37 — corpus.py

A tiny in-memory document set used by the Document Q&A and Summarisation
demos. The documents describe a fictional customer-support SaaS called
NovaDesk. There are NO personal names anywhere — facts only — so the demos
never depend on invented identities (same convention as the S33 corpus).

In a real LangChain app these would come from a document loader
(TextLoader, PyPDFLoader, WebBaseLoader, ...). We keep them inline so the
demos run with zero files on disk and zero network calls.
"""

from __future__ import annotations

from langchain_core.documents import Document


# ---------------------------------------------------------------------
# Raw documents. Each becomes one or more LangChain Document objects
# after splitting. `source` lands in Document.metadata so the Q&A chain
# can cite where an answer came from.
# ---------------------------------------------------------------------

_RAW_DOCS = [
    (
        "overview.md",
        "NovaDesk is a customer-support platform for small and mid-size "
        "software teams. It bundles a shared inbox, a help-centre, and an "
        "AI reply assistant into one product. NovaDesk launched in 2023 "
        "and is used by support teams that want to answer tickets faster "
        "without hiring more agents. The product runs entirely in the "
        "browser and needs no desktop install.",
    ),
    (
        "pricing_and_refunds.md",
        "NovaDesk has three plans. The Starter plan is free for one agent. "
        "The Team plan is billed per agent per month and adds the shared "
        "inbox and basic reporting. The Pro plan adds the AI reply "
        "assistant, custom roles, and the uptime guarantee. Every paid "
        "plan includes a 30-day refund window: if a customer cancels "
        "within 30 days of their first payment, NovaDesk refunds that "
        "payment in full, no questions asked. Refunds are processed back "
        "to the original payment method within five business days.",
    ),
    (
        "sla_and_uptime.md",
        "NovaDesk publishes a service-level agreement for Pro-plan "
        "customers. The SLA guarantees 99.9 percent monthly uptime. If "
        "uptime in a calendar month falls below 99.9 percent, affected "
        "Pro customers receive service credits on the next invoice. "
        "Planned maintenance is announced at least 48 hours in advance "
        "and does not count against the uptime figure. Status and past "
        "incidents are published on the public NovaDesk status page.",
    ),
    (
        "integrations.md",
        "NovaDesk integrates with Slack, Microsoft Teams, and email. The "
        "Slack integration posts new tickets into a chosen channel and "
        "lets agents reply from inside Slack. The Teams integration works "
        "the same way for Microsoft Teams. Email forwarding lets a team "
        "point an existing support address at NovaDesk so every incoming "
        "email becomes a ticket. A REST API is available on the Pro plan "
        "for teams that want to build their own integrations.",
    ),
    (
        "roadmap.md",
        "The NovaDesk roadmap for the current year has three themes. The "
        "first theme is multilingual support, so the AI reply assistant "
        "can draft answers in Spanish, German, and Hindi. The second "
        "theme is deeper reporting, including per-agent response-time "
        "dashboards. The third theme is a mobile app for agents who need "
        "to answer tickets away from their desk. The mobile app is "
        "scheduled for the final quarter of the year.",
    ),
]


def load_documents() -> list[Document]:
    """Return the corpus as LangChain Document objects.

    Stands in for a real document loader. Each Document carries its
    source filename in metadata so answers can be cited.
    """

    return [
        Document(page_content=text, metadata={"source": source})
        for source, text in _RAW_DOCS
    ]


def corpus_as_one_blob() -> str:
    """Concatenate every document into one long string.

    Used by the summarisation demo, which needs a single long input to
    summarise (and to show streaming on a long output).
    """

    return "\n\n".join(text for _, text in _RAW_DOCS)
