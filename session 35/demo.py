"""
Session 35 — demo.py

5 demos for Multi-Agent Systems II (Advanced Architectures).

Run any demo with:
    python demo.py [1|2|3|4|5]

Or run all with:
    python demo.py all

Phase 5 smoke-test command (offline):
    PYTHONPYCACHEPREFIX=/tmp/s35_pycache FAKE_LLM=1 python demo.py all

DEMOS
  1. Hierarchical Manager-of-Managers — Director > 2 Managers > 3 workers.
     Headline demo. Shows Pattern 1 + PROD PATTERN: Hierarchical
     Decomposition + Inter-Agent Message Schema in action.

  2. Debate / Consensus — Bull / Bear / Neutral panelists, Moderator
     synthesises. Shows Pattern 2 + PROD PATTERN: Consensus Synthesis.

  3. Competitive / Best-of-N — three Candidate Writers in parallel,
     Judge picks a winner. Shows Pattern 3 + PROD PATTERN: Best-of-N
     Judging.

  4. Communication Protocol Replay — re-runs Demo 1 and prints the
     full Message bus to show every cross-agent exchange. Pulls the
     PROD PATTERN: Inter-Agent Message Schema into the foreground.

  5. Failure Mode: Escalation — runs the Hierarchical crew with an
     INJECT_BAD_CLAIM payload that will fail fact-checking on every
     revision. Shows MAX_REVISION_ROUNDS firing and the
     EditorialManager ESCALATING to the Director.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Callable, Dict


# Allow `python demo.py` from inside Session_35/Code/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from agent_types import Fact, estimate_call_cost_usd, usd_to_inr
from llm_client import LLMClient
from messages import MessageBus
from trace_logger import banner, section, kv, handle_event, bold
from agents.director import Director
from agents.research_manager import ResearchManager
from agents.editorial_manager import EditorialManager
from agents.debate_panel import DebatePanel
from agents.competitive_panel import CompetitivePanel


def _load_dotenv() -> None:
    """Tiny .env loader so we don't depend on python-dotenv at import
    time. Honours both KEY=VALUE and KEY=\"quoted value\"."""

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
    except OSError:
        pass


def _trace_printer(evt: Dict) -> None:
    handle_event(evt)


# ===========================================================================
# Demo 1 — Hierarchical Manager-of-Managers
# ===========================================================================


def demo_1_hierarchical() -> None:
    banner("DEMO 1 — Hierarchical Manager-of-Managers")
    print(bold("User request: ") + "Write a 3-paragraph brief on RAG.\n")
    print("Topology: Director -> [research_manager, editorial_manager] -> "
          "[3 workers]")
    print("Headline PROD PATTERN: Hierarchical Decomposition.\n")

    llm = LLMClient()
    director = Director(llm, trace=_trace_printer)

    result = director.run("RAG: architecture, evaluation, and quality gates")

    section("FINAL ANSWER")
    print(result.final_answer)

    section("RUN STATS")
    kv("Terminated reason", result.terminated_reason)
    kv("Total LLM calls", result.total_llm_calls)
    kv("Total messages on bus", len(result.messages))


# ===========================================================================
# Demo 2 — Debate / Consensus
# ===========================================================================


def demo_2_debate() -> None:
    banner("DEMO 2 — Debate / Consensus (Bull, Bear, Neutral -> Moderator)")
    topic = "multi-agent systems in production"
    print(bold("Topic: ") + topic + "\n")
    print("Topology: 3 panelists in parallel -> moderator synthesises")
    print("Headline PROD PATTERN: Consensus Synthesis.\n")

    llm = LLMClient()
    panel = DebatePanel(llm, trace=_trace_printer, rounds=1)
    result = panel.run(topic)

    section("CONSENSUS REPORT")
    print(bold("Agreed points:"))
    for p in result.consensus.agreed_points:
        print(f"  - {p}")
    print(bold("\nDisagreements:"))
    for p in result.consensus.disagreements:
        print(f"  - {p}")
    print(bold("\nFinal position:"))
    print(result.consensus.final_position)
    print(bold(f"\nConfidence: {result.consensus.confidence:.2f}"))

    section("RUN STATS")
    kv("Rounds used", result.rounds_used)
    kv("Total LLM calls", result.total_llm_calls)
    kv("Transcript entries", len(result.transcript))


# ===========================================================================
# Demo 3 — Competitive / Best-of-N
# ===========================================================================


def demo_3_competitive() -> None:
    banner("DEMO 3 — Competitive / Best-of-N (3 writers -> Judge)")
    topic = "multi-agent systems"
    print(bold("Topic: ") + topic + "\n")
    print("Topology: 3 candidate writers in parallel -> judge scores")
    print("Headline PROD PATTERN: Best-of-N Judging.\n")

    facts = [
        Fact(text="Multi-agent systems coordinate specialised agents through "
                  "an orchestrator.",
             source="session-34-notes.md"),
        Fact(text="Hierarchical multi-agent systems add a director above the "
                  "orchestrator for two-level decomposition.",
             source="session-35-notes.md"),
        Fact(text="Debate patterns produce a consensus from opposing "
                  "viewpoints rather than picking a winner.",
             source="session-35-notes.md"),
        Fact(text="Competitive best-of-N runs candidates in parallel and "
                  "scores them on explicit criteria.",
             source="session-35-notes.md"),
    ]

    llm = LLMClient()
    panel = CompetitivePanel(llm, trace=_trace_printer)
    result = panel.run(topic, facts)

    section("WINNER")
    kv("Winner ID", result.verdict.winner_id)
    kv("Rationale", result.verdict.rationale)

    section("ALL CANDIDATES")
    for c in result.candidates:
        print(bold(f"\n[{c.candidate_id}] style={c.style}"))
        print(c.draft[:240] + ("..." if len(c.draft) > 240 else ""))

    section("RUN STATS")
    kv("Total LLM calls", result.total_llm_calls)
    kv("Candidates evaluated", len(result.candidates))


# ===========================================================================
# Demo 4 — Communication Protocol Replay
# ===========================================================================


def demo_4_protocol_replay() -> None:
    banner("DEMO 4 — Communication Protocol Replay")
    print(bold("Re-runs the hierarchical demo and prints every Message "
               "on the bus.\n"))
    print("Headline PROD PATTERN: Inter-Agent Message Schema.\n")

    llm = LLMClient()
    director = Director(llm, trace=lambda evt: None)
    result = director.run("RAG: architecture and quality gates")

    section("ALL MESSAGES IN ORDER")
    for i, m in enumerate(result.messages, 1):
        print(f"  {i:3d}. {m.sender:>18s} -> {m.recipient:<20s} "
              f"[{m.intent:<8s}] {m.subject}")

    section("BUS VALIDATION")
    bus = MessageBus()
    for m in result.messages:
        bus.send(m)
    problems = bus.validate()
    _trace_printer({"type": "bus_validation", "problems": problems})

    section("RUN STATS")
    kv("Total messages", len(result.messages))
    kv("Total LLM calls", result.total_llm_calls)
    kv("Terminated reason", result.terminated_reason)

    # Cost math.
    section("COST MATH (illustrative — Haiku 4.5 rates)")
    approx_in_per_call = 500
    approx_out_per_call = 250
    total_in = approx_in_per_call * result.total_llm_calls
    total_out = approx_out_per_call * result.total_llm_calls
    cost_usd = estimate_call_cost_usd(total_in, total_out)
    cost_inr = usd_to_inr(cost_usd)
    kv("Approx input tokens", total_in)
    kv("Approx output tokens", total_out)
    kv("Approx total cost USD", f"${cost_usd:.4f}")
    kv("Approx total cost INR", f"Rs.{cost_inr:.2f}")


# ===========================================================================
# Demo 5 — Failure Mode: Escalation
# ===========================================================================


def demo_5_escalation() -> None:
    banner("DEMO 5 — Failure Mode: Escalation (MAX_REVISION_ROUNDS fires)")
    print(bold("Scenario: ") + "writer keeps planting an ADVERSARIAL_QUESTION "
          "marker that fact-checker will never accept.\n")
    print("Expected: EditorialManager hits MAX_REVISION_ROUNDS=3 and "
          "ESCALATES to the Director.\n")

    # Build an Editorial manager and feed it facts that include the
    # adversarial marker so the writer's canned response includes it.
    llm = LLMClient()
    bus = MessageBus()
    editorial = EditorialManager(llm, bus, trace=_trace_printer)

    facts = [
        Fact(text="ADVERSARIAL_QUESTION is a controlled failure marker for "
                  "this demo.",
             source="session-35-notes.md"),
        Fact(text="ADVERSARIAL_QUESTION can never be accepted by the "
                  "fact-checker.",
             source="session-35-notes.md"),
    ]

    draft, status = editorial.run(facts)

    section("RESULT")
    kv("Status", status)
    kv("Final draft (truncated)", draft[:240] + "...")

    section("RUN STATS")
    kv("Total LLM calls", llm.call_count)
    kv("Messages on bus", len(bus))
    if status == "ESCALATED":
        print(bold("\n[OK] Termination condition fired as expected."))
    else:
        print(bold("\n[unexpected] editorial manager did not escalate"))


# ===========================================================================
# CLI dispatcher
# ===========================================================================


DEMOS = {
    "1": demo_1_hierarchical,
    "2": demo_2_debate,
    "3": demo_3_competitive,
    "4": demo_4_protocol_replay,
    "5": demo_5_escalation,
}


def main() -> None:
    _load_dotenv()

    if len(sys.argv) < 2:
        print(__doc__)
        return

    arg = sys.argv[1].strip().lower()

    if arg == "all":
        for key in ("1", "2", "3", "4", "5"):
            DEMOS[key]()
            print("\n")
        return

    if arg not in DEMOS:
        print(f"Unknown demo '{arg}'. Use 1..5 or 'all'.")
        sys.exit(1)

    DEMOS[arg]()


if __name__ == "__main__":
    main()
