"""
Session 34 — llm_client.py

Thin Anthropic wrapper with FAKE_LLM=1 offline mode for the multi-agent
demos. Every fake route matches on a UNIQUE OPENER-PHRASE from its
system prompt (Phase 5 build convention from S33 — never match on a
topic word).

The fake mode is a deterministic canned router. It's enough to drive
all 5 demos end-to-end without an Anthropic key.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._fake = os.environ.get("FAKE_LLM", "") == "1"
        self._client = None
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def complete(self, system: str, user: str, max_tokens: int = 700) -> str:
        """One-shot text completion. Every agent in S34 uses this entry
        point — we never need decide_next() because the orchestrator's
        JSON output IS the decision."""

        self._call_count += 1
        if self._fake:
            return _fake_one_shot(system, user)
        client = self._get_client()
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts: List[str] = []
        for block in msg.content:
            if getattr(block, "type", "") == "text":
                parts.append(block.text)
        return "".join(parts).strip()

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic SDK not installed. Run 'pip install "
                    "anthropic' or set FAKE_LLM=1."
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client


# ===========================================================================
# FAKE_LLM mode — opener-phrase route dispatch
# ===========================================================================


def _fake_one_shot(system: str, user: str) -> str:
    """Dispatch to the right canned route based on the system prompt's
    opener phrase. See prompts.py — every prompt starts with
    'You are a <role>.' specifically so this dispatch is unambiguous."""

    sys_l = system.lower()

    if "you are an orchestrator" in sys_l:
        return _fake_orchestrator(user)
    if "you are a researcher" in sys_l:
        return _fake_researcher(user)
    if "you are a writer" in sys_l:
        return _fake_writer(user)
    if "you are a fact-checker" in sys_l:
        return _fake_fact_checker(user)
    if "you are a research assistant" in sys_l:
        # The single-agent baseline. Mediocre output — by design.
        return _fake_single_agent_baseline(user)

    return "(fake-llm) no canned route matched"


# ---- orchestrator ----------------------------------------------------------


def _fake_orchestrator(user: str) -> str:
    """The orchestrator reads the scratchpad and emits the next action.

    Decision tree (simple but explicit):
      - no FACTS yet → route to researcher
      - have FACTS, no DRAFT → route to writer
      - have DRAFT, no CRITIQUE → route to fact_checker
      - have CRITIQUE accepted → done, publish FINAL_ANSWER
      - have CRITIQUE rejected → route back to writer for revision
    """

    u = user.lower()
    has_facts = "facts: [{text=" in u
    has_draft = "draft:" in u and "draft: (empty)" not in u
    has_critique_accept = "verdict=accept" in u
    has_critique_reject = "verdict=revise" in u
    user_request = _extract_request(user)

    if has_critique_accept:
        # Synthesise FINAL_ANSWER by pulling the draft from the prompt.
        draft = _extract_draft(user)
        return json.dumps({"done": True, "final": draft})

    if has_critique_reject:
        issues = _extract_issues(user)
        instr = (
            "Revise the draft to address: "
            + "; ".join(issues[:3]) if issues
            else "Revise the draft."
        )
        return json.dumps({"next_worker": "writer", "instruction": instr})

    if not has_facts:
        return json.dumps({
            "next_worker": "researcher",
            "instruction": f"Gather facts on: {user_request}",
        })

    if has_facts and not has_draft:
        return json.dumps({
            "next_worker": "writer",
            "instruction": "Compose a 3-paragraph brief from the facts.",
        })

    if has_draft and not (has_critique_accept or has_critique_reject):
        return json.dumps({
            "next_worker": "fact_checker",
            "instruction": "Verify the draft against the facts.",
        })

    return json.dumps({"done": True, "final": _extract_draft(user) or ""})


# ---- researcher ------------------------------------------------------------


def _fake_researcher(user: str) -> str:
    """Return canned facts based on the topic in the instruction."""

    u = user.lower()
    topic = _extract_topic(user)
    t = topic.lower()

    # Canned fact set 1 — RAG (default-ish)
    if "rag" in t or "retrieval" in t:
        return json.dumps([
            {"text": "RAG stands for Retrieval-Augmented Generation, combining a vector store with a generative model.",
             "source": "rag-overview.md"},
            {"text": "Embeddings convert chunks of text into fixed-length vectors that capture semantic similarity.",
             "source": "embeddings-101.md"},
            {"text": "A vector database such as ChromaDB indexes those vectors for nearest-neighbour search.",
             "source": "chromadb-docs.md"},
            {"text": "Naive RAG performs one retrieval per question; agentic RAG lets an agent decide when and how to retrieve.",
             "source": "session-33-notes.md"},
            {"text": "A quality gate scores retrieved chunks on a 1–5 scale before generation.",
             "source": "session-33-notes.md"},
        ])

    # Canned fact set 2 — Agents
    if "agent" in t or "react" in t or "tool calling" in t:
        return json.dumps([
            {"text": "An LLM agent is a loop that combines model reasoning, tool calls, and tool results.",
             "source": "react-paper.md"},
            {"text": "The ReAct pattern interleaves Thought, Action, and Observation steps each turn.",
             "source": "react-paper.md"},
            {"text": "Native function calling lets the model emit structured tool_use blocks instead of regex-parsed text.",
             "source": "anthropic-docs.md"},
            {"text": "Multi-agent systems coordinate several specialised agents through an orchestrator.",
             "source": "session-34-notes.md"},
        ])

    # Canned fact set 3 — Frameworks (used by parallel demo)
    if "langchain" in t:
        return json.dumps([
            {"text": "LangChain is the earliest popular Python framework for LLM applications, introducing chains and tool agents.",
             "source": "langchain-docs.md"},
            {"text": "Its main abstractions include Runnables, Chains, and Agents.",
             "source": "langchain-docs.md"},
            {"text": "LangChain has been criticised for over-abstraction in early versions; later releases simplified the surface.",
             "source": "langchain-changelog.md"},
        ])
    if "llamaindex" in t:
        return json.dumps([
            {"text": "LlamaIndex specialises in connecting LLMs to private data sources and indexing strategies.",
             "source": "llamaindex-docs.md"},
            {"text": "It offers many index types — vector, list, tree, keyword — for different retrieval shapes.",
             "source": "llamaindex-docs.md"},
            {"text": "LlamaIndex is often paired with LangChain in production RAG pipelines.",
             "source": "llamaindex-docs.md"},
        ])
    if "haystack" in t:
        return json.dumps([
            {"text": "LlamaIndex is often paired with LangChain in production RAG pipelines.",
             "source": "llamaindex-docs.md"},
        ])
    if "haystack" in t:
        return json.dumps([
            {"text": "Haystack is an open-source framework from deepset focused on production NLP and RAG.",
             "source": "haystack-docs.md"},
            {"text": "It uses a pipeline abstraction where components are connected explicitly.",
             "source": "haystack-docs.md"},
            {"text": "Haystack supports both extractive and generative QA out of the box.",
             "source": "haystack-docs.md"},
        ])

    # Default canned facts for any other topic.
    return json.dumps([
        {"text": f"Topic '{topic}' is well-studied in the literature.",
         "source": "general-references.md"},
        {"text": f"Key concepts for '{topic}' include three or four core principles.",
         "source": "general-references.md"},
        {"text": "Production practitioners typically wrap these principles in named patterns.",
         "source": "general-references.md"},
    ])


# ---- writer ----------------------------------------------------------------


def _fake_writer(user: str) -> str:
    u = user.lower()
    is_revision = ("revise" in u and "address" in u) or "issues" in u

    facts = _parse_fact_list(user)
    if not facts:
        return "(fake-writer) no facts provided."

    paragraphs = []

    p1_parts = []
    for fact in facts[:2]:
        text = fact["text"].rstrip(".")
        src = fact.get("source", "unknown")
        p1_parts.append(f"{text} (source: {src}).")
    paragraphs.append(" ".join(p1_parts))

    p2_parts = []
    for fact in facts[2:4]:
        text = fact["text"].rstrip(".")
        src = fact.get("source", "unknown")
        p2_parts.append(f"{text} (source: {src}).")
    if p2_parts:
        paragraphs.append(" ".join(p2_parts))

    if len(facts) > 4:
        p3_parts = []
        for fact in facts[4:]:
            text = fact["text"].rstrip(".")
            src = fact.get("source", "unknown")
            p3_parts.append(f"{text} (source: {src}).")
        paragraphs.append(" ".join(p3_parts))
    else:
        paragraphs.append(
            "Production teams combine these primitives with monitoring, "
            "evaluation, and cost ceilings to make the system reliable."
        )

    if is_revision:
        paragraphs.insert(0, "(Revised draft addressing fact-checker issues.)")

    return "\n\n".join(paragraphs)


# ---- fact-checker ----------------------------------------------------------


def _fake_fact_checker(user: str) -> str:
    draft = _extract_draft(user)
    facts = _parse_fact_list(user)
    fact_sources = {f.get("source", "") for f in facts}

    if "INJECT_BAD_CLAIM" in draft:
        return json.dumps({
            "verdict": "REVISE",
            "issues": ["Claim about 'INJECT_BAD_CLAIM' is not in the fact list."],
            "notes": "Remove the unsupported claim and re-cite the relevant fact."
        })

    if "ADVERSARIAL_QUESTION" in draft:
        return json.dumps({
            "verdict": "REVISE",
            "issues": ["Claim cannot be verified against available facts."],
            "notes": "Insufficient evidence to accept."
        })

    citations = re.findall(r"\(source:\s*([^)]+)\)", draft)
    unknown = [c for c in citations if c.strip() not in fact_sources]
    if unknown:
        return json.dumps({
            "verdict": "REVISE",
            "issues": [f"Unknown source: {c}" for c in unknown[:3]],
            "notes": "Use only sources from the fact list."
        })

    return json.dumps({
        "verdict": "ACCEPT",
        "issues": [],
        "notes": "All claims are cited and the citations match the fact list."
    })


# ---- single-agent baseline -------------------------------------------------


def _fake_single_agent_baseline(user: str) -> str:
    topic = _extract_topic_loose(user)
    return (
        f"{topic.capitalize()} is an important area of study. Several "
        "researchers have written about it (source: general references). "
        "The literature suggests three core ideas that practitioners apply.\n\n"
        f"The mechanisms behind {topic.lower()} involve a number of "
        "well-understood principles. These are taught in introductory "
        "courses and elaborated in more advanced texts (source: textbooks).\n\n"
        "In production, teams combine these ideas with engineering best "
        "practices. The final result is typically deployed as a service "
        "(source: industry blogs)."
    )


# ===========================================================================
# Tiny parsers
# ===========================================================================


def _extract_request(user: str) -> str:
    m = re.search(r"USER_REQUEST:\s*(.+?)(?:\n|$)", user)
    if m:
        return m.group(1).strip()
    return user[:120].strip()


def _extract_topic(user: str) -> str:
    m = re.search(r"on:\s*(.+?)(?:\n|$)", user)
    if m:
        return m.group(1).strip()
    m = re.search(r"about\s+(.+?)(?:[\.\n]|$)", user)
    if m:
        return m.group(1).strip()
    return _extract_request(user)


def _extract_topic_loose(user: str) -> str:
    m = re.search(r"(?:topic|about|on)[:\s]+(.+?)(?:[\.\n]|$)", user, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return user[:60].strip()


def _extract_draft(user: str) -> str:
    m = re.search(r"DRAFT:\s*\n?(.+?)(?:\n[A-Z_]{3,}:|\Z)", user, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _extract_issues(user: str) -> list:
    out = []
    m = re.search(r"issues=(\[.*?\])", user, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    return out


def _parse_fact_list(user: str) -> list:
    # Try the JSON-array form first.
    m = re.search(r"FACTS:\s*\n?(\[[\s\S]*?\])", user)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, list):
                return [
                    {
                        "text": str(f.get("text", "")),
                        "source": str(f.get("source", "")),
                    }
                    for f in parsed
                    if isinstance(f, dict)
                ]
        except json.JSONDecodeError:
            pass
    # Fall back to the fact-checker's "- text=... source=..." form.
    out = []
    for line in user.splitlines():
        mm = re.match(r"\s*-\s*text=\"(.+?)\"\s+source=\"(.+?)\"", line)
        if mm:
            out.append({"text": mm.group(1), "source": mm.group(2)})
    return out
