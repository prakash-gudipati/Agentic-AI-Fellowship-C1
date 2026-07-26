"""
Session 33 — demo.py

Five demos that walk through every decision the agentic RAG loop makes.

Usage:
    python demo.py 1     # WHEN — no retrieval needed (arithmetic)
    python demo.py 2     # WHAT QUERY — naive vs agentic on a vague question
    python demo.py 3     # DECOMPOSE — compound question, two retrievals
    python demo.py 4     # QUALITY GATE — first retrieval misses, re-query
    python demo.py 5     # BUDGET — adversarial question, ceiling hit

By default the demos make REAL Anthropic API calls. Set ANTHROPIC_API_KEY
in your .env. To run offline against the canned fake LLM, prefix with
FAKE_LLM=1:

    FAKE_LLM=1 python demo.py 4

Every demo prints a labelled trace of every retrieval, score, and
decision so the walkthrough script can refer to the lines by name.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ----------------------------------------------------------------------------
# Tiny .env loader (no python-dotenv dependency)
# ----------------------------------------------------------------------------


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).parent / ".env")


from agentic_rag import AgenticRAG  # noqa: E402
from ingest import _get_chroma_collection, ingest  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from naive_rag import answer as naive_answer  # noqa: E402
from trace_logger import (  # noqa: E402
    handle_event,
    print_agentic_summary,
    print_final_answer,
    print_naive_summary,
    print_section,
    print_subheader,
    print_user_question,
)


if os.environ.get("FAKE_LLM", "") != "1" and not os.environ.get(
    "ANTHROPIC_API_KEY"
):
    raise SystemExit(
        "\nERROR: ANTHROPIC_API_KEY is not set in your environment.\n"
        "       Either:\n"
        "         1) export ANTHROPIC_API_KEY=sk-ant-...   (real calls)\n"
        "         2) export FAKE_LLM=1                     (offline mode)\n"
        "       then re-run.\n"
    )


# ----------------------------------------------------------------------------
# Setup — ensure the corpus is ingested before any demo runs
# ----------------------------------------------------------------------------


def _ensure_corpus() -> None:
    coll = _get_chroma_collection(persist=True)
    snap = coll.get()
    if not (snap and snap.get("ids")):
        print_subheader("first run — ingesting corpus into Chroma")
        n = ingest(reset=True)
        print(f"  ingested {n} chunks")


# ----------------------------------------------------------------------------
# Demo 1 — WHEN to retrieve
# ----------------------------------------------------------------------------


def demo_1() -> None:
    print_section("Demo 1 — WHEN to retrieve (the no-retrieval path)")
    print("  Question: arithmetic. A naive RAG would retrieve anyway.")
    print("  Watch the agent decide NOT to call search_kb.")
    _ensure_corpus()

    llm = LLMClient()
    agent = AgenticRAG(llm=llm, on_event=handle_event)
    q = "What is 7 times 9?"
    print_user_question(q)
    ans = agent.answer(q)
    print_final_answer(ans.answer_text)
    print_agentic_summary(ans)


# ----------------------------------------------------------------------------
# Demo 2 — WHAT QUERY (naive vs agentic head-to-head)
# ----------------------------------------------------------------------------


def demo_2() -> None:
    print_section("Demo 2 — Naive RAG vs Agentic RAG on a vague question")
    print(
        "  Same corpus. Same LLM. Same question.\n"
        "  Naive: one shot. Agentic: rewrites the query, then retrieves."
    )
    _ensure_corpus()

    q = "Tell me about your refund stuff."

    print_subheader("Naive RAG run")
    print_user_question(q)
    naive = naive_answer(q)
    print_final_answer(naive.answer_text)
    print_naive_summary(naive)

    print_subheader("Agentic RAG run")
    print_user_question(q)
    agent = AgenticRAG(llm=LLMClient(), on_event=handle_event)
    agentic = agent.answer(q)
    print_final_answer(agentic.answer_text)
    print_agentic_summary(agentic)


# ----------------------------------------------------------------------------
# Demo 3 — DECOMPOSE then retrieve
# ----------------------------------------------------------------------------


def demo_3() -> None:
    print_section("Demo 3 — Decompose-then-Retrieve on a compound question")
    print("  Compound question: TWO independent facts in one sentence.")
    print("  The agent should retrieve TWICE — once per sub-question.")
    _ensure_corpus()

    q = "What primary model does PrepDeck use AND where is the second office located?"
    agent = AgenticRAG(llm=LLMClient(), on_event=handle_event)
    print_user_question(q)
    ans = agent.answer(q)
    print_final_answer(ans.answer_text)
    print_agentic_summary(ans)


# ----------------------------------------------------------------------------
# Demo 4 — QUALITY GATE triggers a re-query
# ----------------------------------------------------------------------------


def demo_4() -> None:
    print_section("Demo 4 — Quality Gate triggers Multi-Hop Retrieval")
    print(
        "  First retrieval pulls chunks that look superficially related\n"
        "  but score below the 3.5 threshold. The agent re-queries with\n"
        "  a sharper formulation and the second retrieval clears the gate."
    )
    _ensure_corpus()

    q = "How long is the on-call shift in Austin?"
    agent = AgenticRAG(
        llm=LLMClient(),
        on_event=handle_event,
        retrieval_budget=3,
    )
    print_user_question(q)
    ans = agent.answer(q)
    print_final_answer(ans.answer_text)
    print_agentic_summary(ans)


# ----------------------------------------------------------------------------
# Demo 5 — Retrieval BUDGET exhausted
# ----------------------------------------------------------------------------


def demo_5() -> None:
    print_section("Demo 5 — Retrieval Budget exhausted on an adversarial query")
    print(
        "  Question: 'Compare Q3 to Q2 in detail'. The agent will keep\n"
        "  retrieving to gather more context. The budget caps it at 3\n"
        "  and forces synthesis from what it has."
    )
    _ensure_corpus()

    q = (
        "Compare the Q3 roadmap to the Q2 roadmap and tell me which "
        "has more items."
    )
    agent = AgenticRAG(
        llm=LLMClient(),
        on_event=handle_event,
        retrieval_budget=3,
        use_decomposition=False,  # so we see the budget enforce by itself
    )
    print_user_question(q)
    ans = agent.answer(q)
    print_final_answer(ans.answer_text)
    print_agentic_summary(ans)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


_DEMOS = {
    "1": demo_1,
    "2": demo_2,
    "3": demo_3,
    "4": demo_4,
    "5": demo_5,
}


def _main(argv) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    key = argv[0].strip()
    fn = _DEMOS.get(key)
    if fn is None:
        print(f"Unknown demo '{key}'. Try one of: {', '.join(_DEMOS)}")
        return 2
    fn()
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
