"""
Session 35 — llm_client.py

Thin Anthropic wrapper with FAKE_LLM=1 offline mode for the hierarchical
+ debate + competitive multi-agent demos.

PHASE 5 BUILD RULE (CLAUDE.md): every fake route matches on a UNIQUE
OPENER-PHRASE from its system prompt — never on a topic word. Every
prompt in prompts.py starts with "You are a <role>." for exactly this
reason.

This file is deliberately long because S35 has more roles than any
prior session — director, two managers, three workers, three debate
panelists, a moderator, candidate writers, and a judge. Each one needs
its own canned route.
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

    def reset_count(self) -> None:
        self._call_count = 0

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        """One-shot text completion. Every agent in S35 uses this entry
        point."""

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
# FAKE_LLM dispatcher — opener-phrase route table
# ===========================================================================


def _fake_one_shot(system: str, user: str) -> str:
    sys_l = system.lower()

    # ----- hierarchical crew --------------------------------------------
    if "you are a director" in sys_l:
        return _fake_director(user)
    if "you are a research_manager" in sys_l:
        return _fake_research_manager(user)
    if "you are an editorial_manager" in sys_l:
        return _fake_editorial_manager(user)
    if "you are a researcher" in sys_l:
        return _fake_researcher(user)
    if "you are a writer" in sys_l:
        return _fake_writer(user)
    if "you are a fact-checker" in sys_l:
        return _fake_fact_checker(user)

    # ----- debate panel --------------------------------------------------
    if "you are a bull_panelist" in sys_l:
        return _fake_bull(user)
    if "you are a bear_panelist" in sys_l:
        return _fake_bear(user)
    if "you are a neutral_panelist" in sys_l:
        return _fake_neutral(user)
    if "you are a moderator" in sys_l:
        return _fake_moderator(user)

    # ----- competitive panel --------------------------------------------
    if "you are a candidate_writer" in sys_l:
        return _fake_candidate(user)
    if "you are a judge" in sys_l:
        return _fake_judge(user)

    return "(fake-llm) no canned route matched"


# ===========================================================================
# Hierarchical fake routes
# ===========================================================================


def _fake_director(user: str) -> str:
    """Director decision tree, mirroring the S34 orchestrator shape but
    one level higher — talks to MANAGERS, not workers."""

    u = user.lower()
    # Use merged_facts_count=N — never substring-match 'merged_facts:'
    # because the label is present even when value is '(none yet)'.
    has_merged_facts = bool(re.search(r"merged_facts_count=([1-9]\d*)", u))
    has_accepted = "editorial_report=accepted" in u
    has_escalation = "editorial_report=escalated" in u
    user_request = _extract_request(user)

    if has_accepted:
        draft = _extract_draft(user)
        return json.dumps({"done": True, "final": draft})

    if has_escalation:
        # Director's fallback when editorial gives up.
        return json.dumps({
            "done": True,
            "final": "(director final) Editorial team escalated; using "
                     "the last reviewed draft as final.\n\n"
                     + _extract_draft(user),
        })

    if not has_merged_facts:
        return json.dumps({
            "next_manager": "research_manager",
            "instruction": f"Gather facts on: {user_request}",
        })

    # We have merged facts but no accept yet — hand to editorial.
    return json.dumps({
        "next_manager": "editorial_manager",
        "instruction": "Draft and review a 3-paragraph brief from the "
                       "merged fact list.",
    })


def _fake_research_manager(user: str) -> str:
    """Research manager splits the request OR merges sub-reports."""

    u = user.lower()
    is_merge = "researcher_reports:" in u

    if is_merge:
        facts = _parse_fact_list(user)
        return json.dumps({
            "merged_facts": facts,
            "report": f"Merged {len(facts)} facts from 2 researchers.",
        })

    topic = _extract_topic(user)
    t = topic.lower()
    # Canned sub-topic splits per topic family.
    if "rag" in t or "retrieval" in t:
        return json.dumps({
            "sub_topics": [
                "RAG architecture and retrieval mechanics",
                "RAG evaluation and quality gates",
            ]
        })
    if "agent" in t or "multi-agent" in t:
        return json.dumps({
            "sub_topics": [
                "Single-agent ReAct loops and tool use",
                "Multi-agent coordination patterns",
            ]
        })
    if "framework" in t or "langchain" in t:
        return json.dumps({
            "sub_topics": [
                "LangChain abstractions and runnables",
                "LlamaIndex indexing strategies",
            ]
        })
    return json.dumps({
        "sub_topics": [
            f"{topic}: foundational concepts",
            f"{topic}: production considerations",
        ]
    })


def _fake_editorial_manager(user: str) -> str:
    """Editorial manager runs the writer-then-fact-checker loop."""

    u = user.lower()
    has_draft = "draft:" in u and "draft: (empty)" not in u
    has_verdict_accept = "verdict=accept" in u
    has_verdict_revise = "verdict=revise" in u
    revision_round = _extract_revision_round(user)

    if has_verdict_accept:
        return json.dumps({
            "action": "REPORT",
            "target": "director",
            "instruction": "Report ACCEPTED draft to director.",
        })

    if has_verdict_revise and revision_round >= 3:
        return json.dumps({
            "action": "REPORT",
            "target": "director",
            "instruction": "Escalate to director — max revisions hit.",
        })

    if has_verdict_revise:
        return json.dumps({
            "action": "REVISE",
            "target": "writer",
            "instruction": "Revise the draft to address fact-checker issues.",
        })

    if has_draft:
        return json.dumps({
            "action": "REVIEW",
            "target": "fact_checker",
            "instruction": "Review the draft against the fact list.",
        })

    return json.dumps({
        "action": "DRAFT",
        "target": "writer",
        "instruction": "Compose a 3-paragraph brief from the fact list.",
    })


def _fake_researcher(user: str) -> str:
    """Reuses S34's canned fact families, keyed by sub-topic phrases."""

    u = user.lower()
    topic = _extract_topic(user)
    t = topic.lower()

    if "rag architecture" in t or ("rag" in t and "retrieval" in t):
        return json.dumps([
            {"text": "RAG stands for Retrieval-Augmented Generation, combining a vector store with a generative model.",
             "source": "rag-overview.md"},
            {"text": "Embeddings convert chunks of text into fixed-length vectors that capture semantic similarity.",
             "source": "embeddings-101.md"},
            {"text": "A vector database such as ChromaDB indexes those vectors for nearest-neighbour search.",
             "source": "chromadb-docs.md"},
            {"text": "Naive RAG performs one retrieval per question; agentic RAG lets an agent decide when and how to retrieve.",
             "source": "session-33-notes.md"},
        ])
    if "rag evaluation" in t or "quality gate" in t:
        return json.dumps([
            {"text": "A quality gate scores retrieved chunks on a 1-5 scale before generation.",
             "source": "session-33-notes.md"},
            {"text": "Ragas measures faithfulness, context relevance, and answer relevance for RAG outputs.",
             "source": "ragas-docs.md"},
            {"text": "Multi-hop retrieval re-queries when the first batch fails the gate.",
             "source": "session-33-notes.md"},
        ])
    if "react" in t or "single-agent" in t or "tool use" in t:
        return json.dumps([
            {"text": "An LLM agent is a loop that combines model reasoning, tool calls, and tool results.",
             "source": "react-paper.md"},
            {"text": "The ReAct pattern interleaves Thought, Action, and Observation steps each turn.",
             "source": "react-paper.md"},
            {"text": "Native function calling lets the model emit structured tool_use blocks instead of regex-parsed text.",
             "source": "anthropic-docs.md"},
        ])
    if "multi-agent coordination" in t or "coordination patterns" in t:
        return json.dumps([
            {"text": "Multi-agent systems coordinate several specialised agents through an orchestrator.",
             "source": "session-34-notes.md"},
            {"text": "Hierarchical multi-agent systems add a director above the orchestrator for two-level decomposition.",
             "source": "session-35-notes.md"},
            {"text": "Debate patterns produce a consensus from opposing viewpoints rather than picking a winner.",
             "source": "session-35-notes.md"},
        ])
    if "langchain" in t or "runnable" in t:
        return json.dumps([
            {"text": "LangChain is the earliest popular Python framework for LLM applications, introducing chains and tool agents.",
             "source": "langchain-docs.md"},
            {"text": "Its main abstractions include Runnables, Chains, and Agents.",
             "source": "langchain-docs.md"},
            {"text": "LangChain has been criticised for over-abstraction in early versions; later releases simplified the surface.",
             "source": "langchain-changelog.md"},
        ])
    if "llamaindex" in t or "indexing" in t:
        return json.dumps([
            {"text": "LlamaIndex specialises in connecting LLMs to private data sources and indexing strategies.",
             "source": "llamaindex-docs.md"},
            {"text": "It offers many index types — vector, list, tree, keyword — for different retrieval shapes.",
             "source": "llamaindex-docs.md"},
            {"text": "LlamaIndex is often paired with LangChain in production RAG pipelines.",
             "source": "llamaindex-docs.md"},
        ])
    # Default — produce three plausible facts so downstream agents have something.
    return json.dumps([
        {"text": f"Topic '{topic}' has three or four widely-cited principles.",
         "source": "general-references.md"},
        {"text": f"Production practitioners typically codify '{topic}' as named patterns.",
         "source": "general-references.md"},
        {"text": f"Trade-offs in '{topic}' depend on workload shape and cost ceilings.",
         "source": "general-references.md"},
    ])


def _fake_writer(user: str) -> str:
    u = user.lower()
    is_revision = ("revise" in u and "address" in u) or "issues" in u
    facts = _parse_fact_list(user)
    if not facts:
        return "(fake-writer) no facts provided."

    paragraphs: List[str] = []
    p1 = " ".join(
        f"{f['text'].rstrip('.')} (source: {f.get('source','unknown')})."
        for f in facts[:2]
    )
    paragraphs.append(p1)

    if len(facts) > 2:
        p2 = " ".join(
            f"{f['text'].rstrip('.')} (source: {f.get('source','unknown')})."
            for f in facts[2:4]
        )
        paragraphs.append(p2)

    if len(facts) > 4:
        p3 = " ".join(
            f"{f['text'].rstrip('.')} (source: {f.get('source','unknown')})."
            for f in facts[4:6]
        )
        paragraphs.append(p3)
    else:
        paragraphs.append(
            "Production teams combine these primitives with monitoring, "
            "evaluation, and cost ceilings to make the system reliable."
        )

    if is_revision:
        paragraphs.insert(0, "(Revised draft addressing fact-checker issues.)")
    return "\n\n".join(paragraphs)


def _fake_fact_checker(user: str) -> str:
    draft = _extract_draft(user)
    facts = _parse_fact_list(user)
    fact_sources = {f.get("source", "") for f in facts}

    if "INJECT_BAD_CLAIM" in draft:
        return json.dumps({
            "verdict": "REVISE",
            "issues": ["Claim about 'INJECT_BAD_CLAIM' is not in the fact list."],
            "notes": "Remove the unsupported claim.",
        })
    if "ADVERSARIAL_QUESTION" in draft:
        return json.dumps({
            "verdict": "REVISE",
            "issues": ["Claim cannot be verified against available facts."],
            "notes": "Insufficient evidence to accept.",
        })

    citations = re.findall(r"\(source:\s*([^)]+)\)", draft)
    unknown = [c for c in citations if c.strip() not in fact_sources]
    if unknown:
        return json.dumps({
            "verdict": "REVISE",
            "issues": [f"Unknown source: {c}" for c in unknown[:3]],
            "notes": "Use only sources from the fact list.",
        })

    return json.dumps({
        "verdict": "ACCEPT",
        "issues": [],
        "notes": "All claims are cited; citations match the fact list.",
    })


# ===========================================================================
# Debate panel fake routes
# ===========================================================================


def _fake_bull(user: str) -> str:
    topic = _extract_topic_loose(user)
    return json.dumps({
        "claim": f"Multi-agent systems will be the dominant production "
                 f"shape for {topic} within two years.",
        "evidence": [
            "Specialisation produces better outputs than a single generalist agent.",
            "Costs are falling fast — coordination overhead is shrinking as a fraction.",
            "Tooling — message buses, observability, judges — is now reusable.",
        ],
    })


def _fake_bear(user: str) -> str:
    topic = _extract_topic_loose(user)
    return json.dumps({
        "claim": f"Multi-agent systems for {topic} will under-deliver "
                 f"versus simpler single-agent baselines in most teams.",
        "evidence": [
            "Coordination overhead grows non-linearly with team size.",
            "Debugging across agent boundaries is materially harder.",
            "Token costs multiply; many teams skip evals and ship blind.",
        ],
    })


def _fake_neutral(user: str) -> str:
    topic = _extract_topic_loose(user)
    return json.dumps({
        "claim": f"Whether multi-agent helps for {topic} depends on three "
                 f"conditions.",
        "evidence": [
            "Bull case wins when sub-tasks are cleanly separable and the team has evals.",
            "Bear case wins when the same task fits in one prompt under 4000 tokens.",
            "Cost math depends on traffic volume and unit economics, not team taste.",
        ],
    })


def _fake_moderator(user: str) -> str:
    return json.dumps({
        "agreed_points": [
            "Specialisation improves output quality when sub-tasks are separable.",
            "Coordination has real overhead — debugging, tokens, latency.",
            "Tooling is improving but is not yet uniform across teams.",
        ],
        "disagreements": [
            "Whether the trend is closer to two years (bull) or five years (bear) for default adoption.",
            "Whether most teams will skip evals and learn the hard way.",
        ],
        "final_position": (
            "Multi-agent systems are the right shape when sub-tasks are "
            "separable, evals are in place, and unit economics support the "
            "extra calls. Single-agent baselines remain the right shape "
            "when the task fits one prompt cleanly. The trend favours "
            "multi-agent, but the timeline is workload-dependent."
        ),
        "confidence": 0.75,
    })


# ===========================================================================
# Competitive panel fake routes
# ===========================================================================


def _fake_candidate(user: str) -> str:
    style = _extract_style(user)
    topic = _extract_topic_loose(user)
    if style == "concise":
        return (
            f"{topic.capitalize()} is best understood as a coordinated "
            "system of specialised agents (source: session-34-notes.md). "
            "Each agent has a narrow prompt and a narrow tool set.\n\n"
            "Production teams add a critique loop and termination "
            "conditions (source: session-34-notes.md).\n\n"
            "The trade-off is cost: more calls, more tokens, more "
            "observability burden (source: session-35-notes.md)."
        )
    if style == "detailed":
        return (
            f"{topic.capitalize()} refers to architectures in which two "
            "or more LLM-driven agents coordinate to answer a single "
            "user request, each with its own system prompt, tool subset, "
            "and output shape (source: session-34-notes.md). The "
            "specialisation produces better task-level outputs than a "
            "single generalist agent (source: session-34-notes.md).\n\n"
            "Production teams apply named patterns — Orchestrator-Worker, "
            "Sequential Pipeline, Parallel/Map, Critique Loop — to keep "
            "the system tractable (source: session-34-notes.md). "
            "Hierarchical decomposition extends this with managerial "
            "layers (source: session-35-notes.md).\n\n"
            "Trade-offs include token cost, coordination overhead, and "
            "debugging difficulty. Mitigations include termination "
            "conditions, structured message schemas, and observability "
            "investments (source: session-35-notes.md)."
        )
    # narrative
    return (
        f"Picture a small consulting firm working on {topic} "
        "(source: session-34-notes.md). A director takes the brief, a "
        "research manager dispatches two analysts, and an editorial "
        "manager runs a writer and a fact-checker. Each conversation is "
        "short. Each role is clear.\n\n"
        "Now picture the same firm without those roles. One generalist "
        "tries to do everything (source: session-34-notes.md). Quality "
        "falls. Costs balloon. The client notices.\n\n"
        "The shape of the firm decides the shape of the output "
        "(source: session-35-notes.md). That's the whole pitch for "
        "hierarchical multi-agent."
    )


def _fake_judge(user: str) -> str:
    """Judge ranks candidates. Canned to favour 'detailed' style for the
    demo, but with sane spread so the rationale is interesting."""

    return json.dumps({
        "winner_id": "candidate_2",
        "rationale": (
            "candidate_2 (detailed) cites the most facts, structures the "
            "trade-offs clearly, and remains readable. candidate_1 "
            "(concise) is sharp but skips production caveats. "
            "candidate_3 (narrative) is engaging but light on evidence."
        ),
        "scores": {
            "candidate_1": {"accuracy": 4, "clarity": 5, "usefulness": 3},
            "candidate_2": {"accuracy": 5, "clarity": 4, "usefulness": 5},
            "candidate_3": {"accuracy": 3, "clarity": 5, "usefulness": 3},
        },
    })


# ===========================================================================
# Tiny parsers — extract structured hints from the user prompt
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
        return m.group(1).strip().strip('"')
    return user[:80].strip()


def _extract_draft(user: str) -> str:
    m = re.search(r"DRAFT:\s*\n?(.+?)(?:\n[A-Z_]{3,}:|\Z)", user, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _extract_style(user: str) -> str:
    m = re.search(r"STYLE:\s*(\w+)", user)
    if m:
        return m.group(1).strip().lower()
    return "detailed"


def _extract_revision_round(user: str) -> int:
    m = re.search(r"REVISION_ROUND:\s*(\d+)", user)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return 0
    return 0


def _parse_fact_list(user: str) -> List[Dict[str, str]]:
    """Pull a JSON-array fact list out of the user prompt. Falls back to
    the bullet form '- text=\"...\" source=\"...\"'."""

    m = re.search(r"FACTS:\s*\n?(\[[\s\S]*?\])", user)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, list):
                return [
                    {"text": str(f.get("text", "")),
                     "source": str(f.get("source", ""))}
                    for f in parsed if isinstance(f, dict)
                ]
        except json.JSONDecodeError:
            pass
    # Researcher reports block — try to merge all JSON arrays inside it.
    m = re.search(r"RESEARCHER_REPORTS:\s*([\s\S]+?)(?:\n[A-Z_]{3,}:|\Z)", user)
    if m:
        out: List[Dict[str, str]] = []
        for jm in re.finditer(r"(\[[\s\S]*?\])", m.group(1)):
            try:
                parsed = json.loads(jm.group(1))
                if isinstance(parsed, list):
                    for f in parsed:
                        if isinstance(f, dict):
                            out.append({
                                "text": str(f.get("text", "")),
                                "source": str(f.get("source", "")),
                            })
            except json.JSONDecodeError:
                continue
        if out:
            return out
    # Fallback - bullet form.
    out2: List[Dict[str, str]] = []
    for line in user.splitlines():
        mm = re.match(r"\s*-\s*text=\"(.+?)\"\s+source=\"(.+?)\"", line)
        if mm:
            out2.append({"text": mm.group(1), "source": mm.group(2)})
    return out2
