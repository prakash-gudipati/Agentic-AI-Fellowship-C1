"""
Session 35 — agents/research_manager.py

PROD PATTERN: Hierarchical Decomposition (level 1 of 2).

The Research Manager is the middle layer between Director and
Researchers. It receives an assignment from the Director, splits it
into TWO sub-topics, dispatches each to one of its researchers, and
merges the two fact lists into a single report.

Role boundary (from prompts.RESEARCH_MANAGER_SYSTEM):
  - split + dispatch + merge ONLY
  - never gather facts itself
  - never write prose

The manager runs the two researchers in parallel via a thread pool —
PROD PATTERN reused from S34's Parallel/Map.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

from agent_types import Fact, Message
from llm_client import LLMClient
from messages import MessageBus
from prompts import RESEARCH_MANAGER_SYSTEM
from agents.researcher import Researcher


class ResearchManager:
    def __init__(self, llm: LLMClient, bus: MessageBus,
                 trace: Callable[[Dict], None],
                 researchers: Optional[List[Researcher]] = None) -> None:
        self.llm = llm
        self.bus = bus
        self.trace = trace
        if researchers is None:
            researchers = [
                Researcher(llm, "R1", trace),
                Researcher(llm, "R2", trace),
            ]
        self.researchers = researchers

    # ------------------------------------------------------------------
    # Step 1 — split the assignment into sub-topics
    # ------------------------------------------------------------------

    def split(self, topic: str, round_num: int = 0) -> List[str]:
        user = f"USER_REQUEST: Split for research.\non: {topic}\n"
        raw = self.llm.complete(RESEARCH_MANAGER_SYSTEM, user)
        sub_topics = _parse_subtopics(raw)
        self.trace({
            "type": "subtopic_split",
            "manager": "research_manager",
            "sub_topics": sub_topics,
        })
        for i, st in enumerate(sub_topics):
            self.bus.assign(
                sender="research_manager",
                recipient=self.researchers[i % len(self.researchers)].researcher_id,
                subject=f"sub-topic {i + 1}",
                payload={"sub_topic": st},
                round_num=round_num,
            )
            self.trace({
                "type": "message_sent",
                "sender": "research_manager",
                "recipient": self.researchers[i % len(self.researchers)].researcher_id,
                "intent": "ASSIGN",
                "subject": f"sub-topic {i + 1}",
            })
        return sub_topics

    # ------------------------------------------------------------------
    # Step 2 — dispatch researchers in parallel
    # ------------------------------------------------------------------

    def dispatch(self, sub_topics: List[str]) -> List[List[Fact]]:
        """Run each researcher on its assigned sub-topic concurrently."""

        if len(sub_topics) != len(self.researchers):
            # Fall back to first-N pairing without crashing.
            pairs = list(zip(self.researchers, sub_topics))
        else:
            pairs = list(zip(self.researchers, sub_topics))

        results: List[List[Fact]] = []
        with ThreadPoolExecutor(max_workers=len(pairs)) as ex:
            futures = [ex.submit(r.run, st) for r, st in pairs]
            for f in futures:
                results.append(f.result())
        return results

    # ------------------------------------------------------------------
    # Step 3 — merge sub-reports into one fact list, then report to
    # the Director.
    # ------------------------------------------------------------------

    def merge_and_report(self, fact_lists: List[List[Fact]],
                         round_num: int = 0) -> List[Fact]:
        # Concatenate; LLM-based merge would dedupe, but the canned data
        # is already deduplicated by sub-topic.
        merged: List[Fact] = []
        seen = set()
        for facts in fact_lists:
            for f in facts:
                key = (f.text, f.source)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(f)

        self.trace({
            "type": "merge_done",
            "manager": "research_manager",
            "num_facts": len(merged),
        })

        # Send a REPORT to the Director.
        self.bus.report(
            sender="research_manager",
            recipient="director",
            subject="merged fact list",
            payload={"merged_facts": [
                {"text": f.text, "source": f.source} for f in merged
            ]},
            round_num=round_num,
        )
        self.trace({
            "type": "message_sent",
            "sender": "research_manager",
            "recipient": "director",
            "intent": "REPORT",
            "subject": "merged fact list",
        })
        return merged

    # ------------------------------------------------------------------
    # End-to-end convenience method.
    # ------------------------------------------------------------------

    def run(self, topic: str, round_num: int = 0) -> List[Fact]:
        sub_topics = self.split(topic, round_num=round_num)
        fact_lists = self.dispatch(sub_topics)
        return self.merge_and_report(fact_lists, round_num=round_num)


def _parse_subtopics(raw: str) -> List[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, dict):
        return []
    val = parsed.get("sub_topics", [])
    if not isinstance(val, list):
        return []
    return [str(x) for x in val][:2]
