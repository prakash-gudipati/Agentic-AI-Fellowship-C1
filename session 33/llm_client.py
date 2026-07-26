"""
Session 33 — llm_client.py

Thin Anthropic wrapper with three modes the rest of the codebase uses:

  complete(system, user) -> str
      One-shot text completion. Used by the query rewriter and the
      quality gate.

  decide_next(system, messages, tools) -> AgentDecision
      The agent-loop entry point. Returns one structured decision:
      either a tool call or a final answer. The agentic RAG loop runs
      a while-loop over this method.

  generate_final(system, user, contexts) -> str
      Naive-RAG-style one-shot answer given a prompt + retrieved chunks.

A FAKE_LLM=1 mode keeps the demos runnable without an API key. The fake
path is intentionally chatty — every branch has a `# fake-route:` comment
so a curious student can read it and see exactly how each demo question
is mocked.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------------
# AgentDecision — what decide_next() returns
# ----------------------------------------------------------------------------


@dataclass
class AgentDecision:
    """One step the LLM tells the agent to take this turn."""

    kind: str  # "tool_call" or "final"
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    text: str = ""
    raw_assistant_blocks: List[Dict[str, Any]] = field(default_factory=list)
    tool_use_id: str = ""


# ----------------------------------------------------------------------------
# LLMClient
# ----------------------------------------------------------------------------


DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._fake = os.environ.get("FAKE_LLM", "") == "1"
        self._client = None

    # ------------------------------------------------------------------
    # One-shot completion — used by rewriter + quality gate
    # ------------------------------------------------------------------

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        if self._fake:
            return _fake_one_shot(system, user)

        client = self._get_client()
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return _flatten_text(msg)

    # ------------------------------------------------------------------
    # decide_next — agent loop entry point
    # ------------------------------------------------------------------

    def decide_next(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 700,
    ) -> AgentDecision:
        if self._fake:
            return _fake_decide(system, messages)

        client = self._get_client()
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )

        # Walk the response blocks. Anthropic returns a mix of text +
        # tool_use blocks. We respect the first tool_use; otherwise the
        # turn is a final answer.
        blocks: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        for block in msg.content:
            t = getattr(block, "type", "")
            if t == "text":
                text_parts.append(block.text)
                blocks.append({"type": "text", "text": block.text})
            elif t == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        for b in blocks:
            if b.get("type") == "tool_use":
                return AgentDecision(
                    kind="tool_call",
                    tool_name=b["name"],
                    tool_args=b.get("input", {}) or {},
                    text="".join(text_parts).strip(),
                    raw_assistant_blocks=blocks,
                    tool_use_id=b["id"],
                )

        return AgentDecision(
            kind="final",
            text="".join(text_parts).strip(),
            raw_assistant_blocks=blocks,
        )

    # ------------------------------------------------------------------
    # generate_final — naive RAG one-shot answer
    # ------------------------------------------------------------------

    def generate_final(
        self,
        system: str,
        user_question: str,
        retrieved_text: str,
        max_tokens: int = 500,
    ) -> str:
        body = (
            "USER QUESTION:\n"
            f"{user_question}\n\n"
            "RETRIEVED CONTEXT:\n"
            f"{retrieved_text}\n\n"
            "Answer the user's question using only the retrieved context. "
            "If the context does not contain the answer, say so plainly."
        )
        return self.complete(system, body, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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


def _flatten_text(message: Any) -> str:
    parts: List[str] = []
    for block in message.content:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
    return "".join(parts).strip()


# ============================================================================
# FAKE_LLM mode — deterministic routes for the five demos
# ============================================================================


def _fake_one_shot(system: str, user: str) -> str:
    """One-shot routes: rewriter, decomposer, quality gate, naive answerer.

    Routes are matched in order. The MORE-SPECIFIC routes come first so a
    system prompt that mentions another component's name in passing (e.g.
    AGENTIC_RAG_SYSTEM contains the phrase "quality gate" because it tells
    the agent to react when the gate fails) doesn't poach the wrong route.
    """

    sys_l = system.lower()
    usr_l = user.lower()

    # ----- Query rewriter (matches the rewriter system prompt) -----
    # fake-route: query rewriter — return a tightened query string.
    if "you are a query rewriter" in sys_l:
        raw = _find_after(user, "USER_QUESTION:")
        return _fake_rewrite(raw)

    # ----- Decomposer (matches the decomposer system prompt) -----
    # fake-route: decomposer — returns JSON array of sub-questions.
    if "you are a question decomposer" in sys_l:
        raw = _find_after(user, "USER_QUESTION:")
        subs = _fake_decompose(raw)
        return json.dumps(subs)

    # ----- Quality gate (matches the relevance-scorer system prompt) -----
    # fake-route: relevance scorer — token-overlap heuristic.
    if "you are a relevance scorer" in sys_l:
        return _fake_score_chunks(user)

    # ----- Naive RAG answerer -----
    # fake-route: naive answer — pull verbatim sentences that overlap
    # the user question's keywords. Deliberately dumb to make the
    # naive-vs-agentic comparison honest.
    if "naive rag" in sys_l:
        return _fake_naive_answer(user)

    # ----- Agentic-RAG synthesis (called by the final synthesise pass) -----
    if "you are an agent answering questions" in sys_l:
        return _fake_agentic_final(user)

    # Fallback so the FAKE path is never silent.
    return "(fake-llm) no canned route matched — system prompt unfamiliar."


def _fake_decide(system: str, messages: List[Dict[str, Any]]) -> AgentDecision:
    """Decide one step in the agentic loop.

    The fake reads the most recent user message and the tool-result
    history, then picks one of:
      - emit a search_kb tool call (with a tightened query)
      - emit a final answer

    Each demo question maps to a hand-traced sequence below.
    """

    # The original user question is always the first user message.
    first_user = ""
    last_user = ""
    tool_calls_so_far: List[Dict[str, Any]] = []
    tool_results_so_far: List[str] = []

    for m in messages:
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):
                # tool_result blocks
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "tool_result":
                        tool_results_so_far.append(
                            str(blk.get("content", ""))
                        )
            elif isinstance(content, str):
                if not first_user:
                    first_user = content
                last_user = content
        elif m.get("role") == "assistant":
            content = m.get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "tool_use":
                        tool_calls_so_far.append(blk)

    q = (first_user or last_user).strip()
    q_l = q.lower()
    attempts_done = len(tool_calls_so_far)

    # ---- Decision tree, demo by demo ----

    # Demo 1 — arithmetic / no-retrieval branch
    # fake-route: detect a maths question → answer directly.
    if _is_arithmetic(q_l):
        return AgentDecision(
            kind="final",
            text=_fake_arithmetic_answer(q),
        )

    # Demo 1b — general-knowledge no-retrieval branch
    if "capital of india" in q_l or "capital of india?" in q_l:
        return AgentDecision(
            kind="final",
            text="The capital of India is New Delhi. (No retrieval needed.)",
        )

    # Demo 2 — poorly-phrased refund question, single retrieval
    if "refund" in q_l and attempts_done == 0:
        return AgentDecision(
            kind="tool_call",
            tool_name="search_kb",
            tool_args={"query": "refund policy and refund window", "k": 4},
            tool_use_id=f"call_{attempts_done+1}",
        )

    # Demo 4 — Austin on-call, may need a re-query if quality gate fails
    if "austin" in q_l and "on-call" in q_l:
        if attempts_done == 0:
            return AgentDecision(
                kind="tool_call",
                tool_name="search_kb",
                tool_args={"query": "Austin engineering office", "k": 3},
                tool_use_id=f"call_{attempts_done+1}",
            )
        if attempts_done == 1 and _last_gate_failed(tool_results_so_far):
            return AgentDecision(
                kind="tool_call",
                tool_name="search_kb",
                tool_args={
                    "query": "on-call rotation primary shift hours",
                    "k": 4,
                },
                tool_use_id=f"call_{attempts_done+1}",
            )

    # Demo 3 — compound question: "What model AND where is the second office"
    if "model" in q_l and ("office" in q_l or "headquarter" in q_l):
        if attempts_done == 0:
            return AgentDecision(
                kind="tool_call",
                tool_name="search_kb",
                tool_args={
                    "query": "primary reasoning model for code review",
                    "k": 3,
                },
                tool_use_id=f"call_{attempts_done+1}",
            )
        if attempts_done == 1:
            return AgentDecision(
                kind="tool_call",
                tool_name="search_kb",
                tool_args={"query": "second office location Austin", "k": 3},
                tool_use_id=f"call_{attempts_done+1}",
            )

    # Demo 5 — adversarial: roadmap comparison, exhausts budget
    if "roadmap" in q_l or ("q3" in q_l and "q2" in q_l):
        if attempts_done == 0:
            return AgentDecision(
                kind="tool_call",
                tool_name="search_kb",
                tool_args={"query": "Q2 2026 roadmap in progress", "k": 4},
                tool_use_id=f"call_{attempts_done+1}",
            )
        if attempts_done == 1:
            return AgentDecision(
                kind="tool_call",
                tool_name="search_kb",
                tool_args={"query": "Q3 2026 roadmap planned items", "k": 4},
                tool_use_id=f"call_{attempts_done+1}",
            )
        if attempts_done == 2:
            return AgentDecision(
                kind="tool_call",
                tool_name="search_kb",
                tool_args={
                    "query": "Q4 2026 roadmap exploratory mobile launch",
                    "k": 4,
                },
                tool_use_id=f"call_{attempts_done+1}",
            )

    # Default first-retrieval branch for any other question
    if attempts_done == 0:
        return AgentDecision(
            kind="tool_call",
            tool_name="search_kb",
            tool_args={"query": q, "k": 4},
            tool_use_id="call_1",
        )

    # All retrievals exhausted → synthesise final answer.
    return AgentDecision(
        kind="final",
        text=_fake_synthesise_final(q, tool_results_so_far),
    )


# ----------------------------------------------------------------------------
# Helpers for the fake routes
# ----------------------------------------------------------------------------


_ARITHMETIC_RE = re.compile(
    r"\b(\d+)\s*([\+\-\*x×÷/])\s*(\d+)\b"
)


def _is_arithmetic(text: str) -> bool:
    if _ARITHMETIC_RE.search(text):
        return True
    if "times" in text and any(d in text for d in "0123456789"):
        return True
    return False


def _fake_arithmetic_answer(q: str) -> str:
    m = _ARITHMETIC_RE.search(q.replace("×", "*").replace("÷", "/"))
    if m:
        a, op, b = m.group(1), m.group(2), m.group(3)
        x, y = int(a), int(b)
        if op == "+":
            result = x + y
        elif op == "-":
            result = x - y
        elif op in ("*", "x"):
            result = x * y
        elif op == "/":
            result = x // y if y else 0
        else:
            return "(fake) couldn't parse the arithmetic."
        return f"{a} {op} {b} = {result}. (No retrieval needed — this is "
        "general arithmetic.)"
    # "what's 7 times 9"
    nums = [int(n) for n in re.findall(r"\d+", q)]
    if len(nums) == 2 and "times" in q.lower():
        return (
            f"{nums[0]} times {nums[1]} = {nums[0] * nums[1]}. "
            "(No retrieval needed — this is general arithmetic.)"
        )
    return "(fake) arithmetic detected but unparseable."


def _fake_rewrite(raw: str) -> str:
    """Return a tightened version of the user's raw query."""

    r = raw.strip().lower()
    if "refund" in r:
        return "refund policy and refund window"
    if "on-call" in r or "oncall" in r:
        return "on-call rotation primary shift hours"
    if "model" in r and ("office" in r or "headquarter" in r):
        return "primary reasoning model for code review"
    if "pro plan" in r:
        return "Pro plan pricing and features"
    if "ai mentor" in r:
        return "AI mentor model and stack"
    if "headquarter" in r or "office" in r:
        return "PrepDeck office locations Bengaluru Austin"
    # Default: strip filler words.
    stop = {"tell", "me", "about", "your", "the", "stuff", "thing", "please", "ok"}
    keep = [t for t in re.split(r"\s+", raw.strip()) if t.lower() not in stop]
    return " ".join(keep) if keep else raw.strip()


def _fake_decompose(raw: str) -> List[str]:
    # The user prompt embeds both the question AND an instruction. Take
    # only the first non-empty line — that's the question.
    first_line = ""
    for line in raw.splitlines():
        if line.strip():
            first_line = line.strip()
            break
    if not first_line:
        first_line = raw.strip()

    r = first_line.lower()
    if " and " in r:
        parts = [
            p.strip(" ?.,;")
            for p in re.split(
                r"\s+and\s+", first_line, maxsplit=1, flags=re.IGNORECASE
            )
        ]
        if len(parts) == 2 and all(parts):
            return parts
    if "compare" in r and "q3" in r and "q2" in r:
        return [
            "what items are in the Q2 2026 roadmap",
            "what items are in the Q3 2026 roadmap",
        ]
    return [first_line]


def _fake_score_chunks(user_prompt: str) -> str:
    """Score each chunk 1..5 based on keyword overlap with the query.

    Output format expected by the quality gate parser:
        SCORE chunk_id=<id> relevance=<1..5> reason=<short reason>
        ...
    """

    query = _find_after(user_prompt, "QUERY:")
    chunks_block = _find_after(user_prompt, "CHUNKS:")
    if not chunks_block:
        return ""
    q_tokens = _keywords(query)

    out_lines: List[str] = []
    for raw in chunks_block.splitlines():
        line = raw.strip()
        if not line.startswith("- chunk_id="):
            continue
        m = re.match(
            r"- chunk_id=(\S+)\s+source=(\S+)\s+text=(.+)$",
            line,
        )
        if not m:
            continue
        cid, src, text = m.group(1), m.group(2), m.group(3)
        c_tokens = _keywords(text)
        overlap = q_tokens & c_tokens
        # Score boosting: certain keywords gate the chunk to 4/5.
        score = 1
        if overlap:
            score = min(5, 1 + len(overlap))
        # Source-aware boost so the quality gate fires correctly on
        # the on-call demo: a chunk whose source matches one of the
        # query's keywords gets +1.
        src_token = src.split("_", 1)[-1].split(".", 1)[0]
        if src_token in q_tokens:
            score = min(5, score + 1)
        # Penalty: very short chunks are usually wrong.
        if len(text) < 80:
            score = max(1, score - 1)
        out_lines.append(
            f"SCORE chunk_id={cid} relevance={score} "
            f"reason=overlap({len(overlap)})"
        )
    return "\n".join(out_lines)


def _fake_naive_answer(user_prompt: str) -> str:
    """The naive baseline. Quotes the most overlapping sentence verbatim."""

    question = _find_after(user_prompt, "USER QUESTION:")
    context = _find_after(user_prompt, "RETRIEVED CONTEXT:")
    if not context:
        return "I don't have any context to answer from."
    q_tokens = _keywords(question)
    best = (-1, "")
    for sent in re.split(r"(?<=[\.\!\?])\s+", context):
        s_tokens = _keywords(sent)
        overlap = len(q_tokens & s_tokens)
        if overlap > best[0]:
            best = (overlap, sent.strip())
    if best[1]:
        return f"(naive) Based on the retrieved chunks: {best[1]}"
    return "(naive) The retrieved chunks did not seem to answer the question."


def _fake_agentic_final(user_prompt: str) -> str:
    """Synthesise the final answer from a SYNTHESIS prompt the loop builds."""

    question = _find_after(user_prompt, "USER_QUESTION:")
    evidence = _find_after(user_prompt, "EVIDENCE:")
    if not evidence:
        return "I could not retrieve enough information to answer."
    q_l = question.lower()
    e_l = evidence.lower()

    if "refund" in q_l:
        if "14 days" in e_l:
            return (
                "PrepDeck offers a full refund within 14 days of purchase, "
                "no questions asked. After 14 days, refunds are case-by-case "
                "and require a written reason. Annual customers get a "
                "prorated refund minus a 10% admin fee. Source: 02_pricing.md."
            )
    if "model" in q_l and ("office" in q_l or "headquarter" in q_l):
        model_part = "Claude Haiku 4.5"
        office_part = "a second office in Austin, Texas, opened in November 2025"
        if "claude haiku 4.5" in e_l:
            model_part = "Claude Haiku 4.5"
        if "austin" in e_l:
            office_part = "a second office in Austin, Texas, opened in November 2025"
        return (
            f"PrepDeck uses {model_part} as its primary reasoning model for "
            f"code review and mock interviews. The company is headquartered in "
            f"Bengaluru, India, with {office_part}. "
            "Sources: 05_ai_stack.md, 01_company_overview.md."
        )
    if "austin" in q_l and "on-call" in q_l:
        if "09:00 cst" in e_l or "21:00 cst" in e_l:
            return (
                "Austin engineers cover the same 7-day on-call blocks as the "
                "Bengaluru rotation, but their primary shift hours are "
                "09:00 CST to 21:00 CST. Source: 03_engineering_handbook.md."
            )
    if "q3" in q_l and "q2" in q_l:
        return (
            "Q2 in-progress items: iOS app launch, behavioural-interview "
            "track, group study rooms, Hindi-language UI. Q3 planned items: "
            "US mobile launch, live cohort feature, spaced repetition, "
            "LeetCode/HackerRank integration. Both quarters list four major "
            "items; Q2 items are in active development, Q3 items are "
            "planned. Source: 04_product_roadmap.md."
        )

    return (
        "I've gathered some information but cannot synthesise a fluent "
        "answer without the real LLM. The evidence is: "
        f"{evidence[:300]}..."
    )


def _fake_synthesise_final(question: str, results: List[str]) -> str:
    evidence_blob = "\n".join(results)
    prompt = f"USER_QUESTION:\n{question}\nEVIDENCE:\n{evidence_blob}\n"
    return _fake_agentic_final(prompt)


def _last_gate_failed(tool_results: List[str]) -> bool:
    """Look at the most recent tool result and check whether the gate text
    reported a failure."""

    if not tool_results:
        return False
    last = tool_results[-1].lower()
    return "gate: fail" in last or "average relevance" in last and "below 3.5" in last


def _keywords(text: str) -> set:
    out = set()
    for tok in re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower()):
        if len(tok) > 3:
            out.add(tok)
    return out


def _find_after(text: str, marker: str) -> str:
    idx = text.find(marker)
    if idx < 0:
        return ""
    tail = text[idx + len(marker) :]
    # Stop at the next ALLCAPS_HEADER: or end of string.
    m = re.search(r"\n[A-Z_]{3,}:", tail)
    if m:
        return tail[: m.start()].strip()
    return tail.strip()
