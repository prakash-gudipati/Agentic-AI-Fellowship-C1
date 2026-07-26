"""
Session 34 — demo.py

Five demos that walk through the multi-agent patterns introduced in
the session.

Usage:
    python demo.py 1     # Single agent vs Crew on same question
    python demo.py 2     # Sequential Pipeline (Researcher then Writer)
    python demo.py 3     # Parallel / Map (3 researchers fan-out)
    python demo.py 4     # Critique Loop (Writer + Fact-Checker)
    python demo.py 5     # Termination ceiling (adversarial)

Recommended smoke-test command (Phase 5 build convention):
    PYTHONPYCACHEPREFIX=/tmp/s34_pycache FAKE_LLM=1 python demo.py 1
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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


from agents.fact_checker import FactChecker  # noqa: E402
from agents.orchestrator import Orchestrator  # noqa: E402
from agents.researcher import Researcher  # noqa: E402
from agents.writer import Writer  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from scratchpad import Scratchpad  # noqa: E402
from single_agent_baseline import answer as single_agent_answer  # noqa: E402
from trace_logger import (  # noqa: E402
    handle_event,
    print_final_answer,
    print_scratchpad,
    print_section,
    print_subheader,
    print_user_request,
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
# Demo 1 — Single Agent vs Crew on the same question
# ----------------------------------------------------------------------------


def demo_1() -> None:
    print_section("Demo 1 — Single Agent vs Crew on the same question")
    print(
        "  Same model. Same task. Single agent does it in one big prompt;\n"
        "  the crew specialises by role."
    )
    q = "Write a 3-paragraph brief on Retrieval-Augmented Generation."

    print_subheader("SINGLE-AGENT BASELINE")
    print_user_request(q)
    baseline = single_agent_answer(q, llm=LLMClient())
    print_final_answer(baseline.final_text)
    print(f"\n  llm_calls={baseline.llm_calls}  time={baseline.elapsed_seconds:.2f}s")

    print_subheader("MULTI-AGENT CREW")
    print_user_request(q)
    crew = Orchestrator(llm=LLMClient(), on_event=handle_event)
    result = crew.run(q)
    print_final_answer(result.final_text)
    print(f"\n  llm_calls={result.llm_calls}  rounds={result.rounds}  "
          f"reason={result.terminated_reason}  time={result.elapsed_seconds:.2f}s")


# ----------------------------------------------------------------------------
# Demo 2 — Sequential Pipeline (no orchestrator, fixed order)
# ----------------------------------------------------------------------------


def demo_2() -> None:
    print_section("Demo 2 — Sequential Pipeline: Researcher → Writer")
    print(
        "  No orchestrator. Fixed order. Researcher's output IS the\n"
        "  writer's input. Two LLM calls. Watch the scratchpad fill up."
    )
    q = "Multi-Agent Systems"

    print_user_request(f"Brief me on: {q}")
    llm = LLMClient()
    sp = Scratchpad()

    researcher = Researcher(llm=llm)
    writer = Writer(llm=llm)

    handle_event("round_start", {"round_no": 1})
    handle_event("worker_started", {"worker": "researcher"})
    facts = researcher.run(f"Gather facts on: {q}", sp, round_no=1)
    handle_event("worker_finished", {"worker": "researcher", "summary": f"{len(facts)} facts"})

    handle_event("round_start", {"round_no": 2})
    handle_event("worker_started", {"worker": "writer"})
    draft = writer.run(
        "Compose a 3-paragraph brief from the facts.", sp, round_no=2
    )
    handle_event("worker_finished", {"worker": "writer", "summary": f"draft {len(draft)} chars"})

    print_final_answer(draft)
    print_scratchpad("after pipeline:", sp.summary())
    print(f"\n  llm_calls={llm.call_count}  time=(fixed sequential, 2 calls)")


# ----------------------------------------------------------------------------
# Demo 3 — Parallel / Map (3 researchers fan-out + aggregator)
# ----------------------------------------------------------------------------


def demo_3() -> None:
    print_section("Demo 3 — Parallel / Map: 3 researchers fan-out")
    print(
        "  Three independent sub-topics. Three workers in parallel.\n"
        "  Aggregator merges the results."
    )
    frameworks = ["LangChain", "LlamaIndex", "Haystack"]
    print_user_request(
        f"Compare these frameworks: {', '.join(frameworks)}"
    )

    llm = LLMClient()
    pads = [Scratchpad() for _ in frameworks]

    def run_one(idx_topic):
        idx, topic = idx_topic
        researcher = Researcher(llm=llm)
        return idx, topic, researcher.run(
            f"Gather facts on: {topic}", pads[idx], round_no=1
        )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run_one, enumerate(frameworks)))
    elapsed = time.perf_counter() - started

    handle_event("round_start", {"round_no": 1})
    for idx, topic, facts in results:
        handle_event(
            "worker_finished",
            {"worker": f"researcher[{topic}]", "summary": f"{len(facts)} facts"},
        )

    # Aggregator — flatten into one summary.
    handle_event("round_start", {"round_no": 2})
    handle_event("worker_started", {"worker": "aggregator"})
    paragraphs = []
    for _, topic, facts in results:
        head = f"{topic}: " + " ".join(f.text for f in facts[:2])
        paragraphs.append(head)
    final = "\n\n".join(paragraphs)
    handle_event("worker_finished", {"worker": "aggregator", "summary": f"merged {len(results)} sub-reports"})

    print_final_answer(final)
    print(f"\n  llm_calls={llm.call_count}  wall_time={elapsed:.2f}s "
          "(parallel — sequential would be ~3x this)")


# ----------------------------------------------------------------------------
# Demo 4 — Critique Loop: Writer + Fact-Checker with one revision cycle
# ----------------------------------------------------------------------------


def demo_4() -> None:
    print_section("Demo 4 — Critique Loop: writer + fact-checker")
    print(
        "  Writer plants a bad claim on round 1 (demo marker).\n"
        "  Fact-Checker catches it. Writer revises. Fact-Checker accepts.\n"
        "  Termination on quality_met."
    )
    q = "Write a 3-paragraph brief on agents and tool calling."
    print_user_request(q)

    crew = Orchestrator(
        llm=LLMClient(),
        on_event=handle_event,
        writer_inject_marker="INJECT_BAD_CLAIM",
        max_rounds=5,
    )
    result = crew.run(q)
    print_final_answer(result.final_text)
    print(f"\n  llm_calls={result.llm_calls}  rounds={result.rounds}  "
          f"reason={result.terminated_reason}  time={result.elapsed_seconds:.2f}s")


# ----------------------------------------------------------------------------
# Demo 5 — Termination ceiling on an adversarial question
# ----------------------------------------------------------------------------


def demo_5() -> None:
    print_section("Demo 5 — Termination: budget exhausted on adversarial Q")
    print(
        "  Writer plants an 'ADVERSARIAL_QUESTION' marker every round.\n"
        "  Fact-Checker never accepts. Crew hits max_rounds=3 and ships\n"
        "  the best draft with a flag."
    )
    q = "Write a 3-paragraph brief on a topic with no source available."
    print_user_request(q)

    crew = Orchestrator(
        llm=LLMClient(),
        on_event=handle_event,
        writer_inject_marker="ADVERSARIAL_QUESTION",
        max_rounds=3,
    )
    result = crew.run(q)
    print_final_answer(result.final_text)
    print(f"\n  llm_calls={result.llm_calls}  rounds={result.rounds}  "
          f"reason={result.terminated_reason}  time={result.elapsed_seconds:.2f}s")


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


# ----------------------------------------------------------------------------
# Demo 2 — Sequential Pipeline (no orchestrator, fixed order)
# ----------------------------------------------------------------------------


def demo_2() -> None:
    print_section("Demo 2 — Sequential Pipeline: Researcher then Writer")
    print(
        "  No orchestrator. Fixed order. Researcher's output IS the\n"
        "  writer's input. Two LLM calls. Watch the scratchpad fill up."
    )
    q = "Multi-Agent Systems"

    print_user_request(f"Brief me on: {q}")
    llm = LLMClient()
    sp = Scratchpad()

    researcher = Researcher(llm=llm)
    writer = Writer(llm=llm)

    handle_event("round_start", {"round_no": 1})
    handle_event("worker_started", {"worker": "researcher"})
    facts = researcher.run(f"Gather facts on: {q}", sp, round_no=1)
    handle_event("worker_finished", {"worker": "researcher", "summary": f"{len(facts)} facts"})

    handle_event("round_start", {"round_no": 2})
    handle_event("worker_started", {"worker": "writer"})
    draft = writer.run(
        "Compose a 3-paragraph brief from the facts.", sp, round_no=2
    )
    handle_event("worker_finished", {"worker": "writer", "summary": f"draft {len(draft)} chars"})

    print_final_answer(draft)
    print_scratchpad("after pipeline:", sp.summary())
    print(f"\n  llm_calls={llm.call_count}  time=(fixed sequential, 2 calls)")


# ----------------------------------------------------------------------------
# Demo 3 — Parallel / Map
# ----------------------------------------------------------------------------


def demo_3() -> None:
    print_section("Demo 3 — Parallel / Map: 3 researchers fan-out")
    print(
        "  Three independent sub-topics. Three workers in parallel.\n"
        "  Aggregator merges the results."
    )
    frameworks = ["LangChain", "LlamaIndex", "Haystack"]
    print_user_request(f"Compare these frameworks: {', '.join(frameworks)}")

    llm = LLMClient()
    pads = [Scratchpad() for _ in frameworks]

    def run_one(idx_topic):
        idx, topic = idx_topic
        researcher = Researcher(llm=llm)
        return idx, topic, researcher.run(
            f"Gather facts on: {topic}", pads[idx], round_no=1
        )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run_one, enumerate(frameworks)))
    elapsed = time.perf_counter() - started

    handle_event("round_start", {"round_no": 1})
    for idx, topic, facts in results:
        handle_event(
            "worker_finished",
            {"worker": f"researcher[{topic}]", "summary": f"{len(facts)} facts"},
        )

    handle_event("round_start", {"round_no": 2})
    handle_event("worker_started", {"worker": "aggregator"})
    paragraphs = []
    for _, topic, facts in results:
        head = f"{topic}: " + " ".join(f.text for f in facts[:2])
        paragraphs.append(head)
    final = "\n\n".join(paragraphs)
    handle_event("worker_finished", {"worker": "aggregator", "summary": f"merged {len(results)} sub-reports"})

    print_final_answer(final)
    print(f"\n  llm_calls={llm.call_count}  wall_time={elapsed:.2f}s "
          "(parallel)")


# ----------------------------------------------------------------------------
# Demo 4 — Critique Loop
# ----------------------------------------------------------------------------


def demo_4() -> None:
    print_section("Demo 4 — Critique Loop: writer + fact-checker")
    print(
        "  Writer plants a bad claim on round 1 (demo marker).\n"
        "  Fact-Checker catches it. Writer revises. Fact-Checker accepts."
    )
    q = "Write a 3-paragraph brief on agents and tool calling."
    print_user_request(q)

    crew = Orchestrator(
        llm=LLMClient(),
        on_event=handle_event,
        writer_inject_marker="INJECT_BAD_CLAIM",
        max_rounds=5,
    )
    result = crew.run(q)
    print_final_answer(result.final_text)
    print(f"\n  llm_calls={result.llm_calls}  rounds={result.rounds}  "
          f"reason={result.terminated_reason}  time={result.elapsed_seconds:.2f}s")


# ----------------------------------------------------------------------------
# Demo 5 — Termination ceiling
# ----------------------------------------------------------------------------


def demo_5() -> None:
    print_section("Demo 5 — Termination: budget exhausted on adversarial Q")
    print(
        "  Writer plants an 'ADVERSARIAL_QUESTION' marker every round.\n"
        "  Fact-Checker never accepts. Crew hits max_rounds=3 and ships\n"
        "  the best draft with a flag."
    )
    q = "Write a 3-paragraph brief on a topic with no source available."
    print_user_request(q)

    crew = Orchestrator(
        llm=LLMClient(),
        on_event=handle_event,
        writer_inject_marker="ADVERSARIAL_QUESTION",
        max_rounds=3,
    )
    result = crew.run(q)
    print_final_answer(result.final_text)
    print(f"\n  llm_calls={result.llm_calls}  rounds={result.rounds}  "
          f"reason={result.terminated_reason}  time={result.elapsed_seconds:.2f}s")


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
