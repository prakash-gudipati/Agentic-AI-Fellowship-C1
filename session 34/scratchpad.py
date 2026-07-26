"""
Session 34 — scratchpad.py

PROD PATTERN: Shared Scratchpad.

Typed structured memory that every agent in the crew can read and write.
Not free-form chat history.

Named sections — FACTS, DRAFT, CRITIQUE, FINAL_ANSWER. Each section has
an owner agent and a list of writers. Reads are targeted. Every write
is labelled. This is S32's Context Builder lesson applied across
agents instead of across turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_types import Critique, Fact


# Named sections — these are the ONLY places agents can write.
SECTION_FACTS = "FACTS"
SECTION_DRAFT = "DRAFT"
SECTION_CRITIQUE = "CRITIQUE"
SECTION_FINAL = "FINAL_ANSWER"
SECTION_NOTES = "NOTES"   # free-form, low-priority log


_ALLOWED_SECTIONS = {
    SECTION_FACTS,
    SECTION_DRAFT,
    SECTION_CRITIQUE,
    SECTION_FINAL,
    SECTION_NOTES,
}


@dataclass
class ScratchpadEvent:
    """One write event. The trace logger reads these."""

    round_no: int
    agent: str
    section: str
    summary: str   # one-line summary; full value lives in the section


@dataclass
class Scratchpad:
    """The shared memory.

    Each section maps to ONE value (str, list, dict, or domain object).
    Writes overwrite — agents must read first if they want to append.
    The event log is append-only and the trace logger walks it.
    """

    sections: Dict[str, Any] = field(default_factory=dict)
    events: List[ScratchpadEvent] = field(default_factory=list)

    # --------------------------------------------------------------
    # Writes
    # --------------------------------------------------------------

    def write(
        self,
        section: str,
        value: Any,
        *,
        agent: str,
        round_no: int,
        summary: str = "",
    ) -> None:
        if section not in _ALLOWED_SECTIONS:
            raise ValueError(
                f"unknown scratchpad section '{section}'. "
                f"Allowed: {sorted(_ALLOWED_SECTIONS)}"
            )
        self.sections[section] = value
        self.events.append(
            ScratchpadEvent(
                round_no=round_no,
                agent=agent,
                section=section,
                summary=summary or _summarise(value),
            )
        )

    def append_to_notes(
        self, line: str, *, agent: str, round_no: int
    ) -> None:
        existing = self.sections.get(SECTION_NOTES, [])
        if not isinstance(existing, list):
            existing = [str(existing)]
        existing.append(f"[{agent}] {line}")
        self.sections[SECTION_NOTES] = existing
        self.events.append(
            ScratchpadEvent(
                round_no=round_no,
                agent=agent,
                section=SECTION_NOTES,
                summary=line[:60],
            )
        )

    # --------------------------------------------------------------
    # Reads (targeted, with sensible default)
    # --------------------------------------------------------------

    def read(self, section: str, default: Any = None) -> Any:
        if section not in _ALLOWED_SECTIONS:
            raise ValueError(f"unknown section '{section}'")
        return self.sections.get(section, default)

    def get_facts(self) -> List[Fact]:
        return list(self.sections.get(SECTION_FACTS, []) or [])

    def get_draft(self) -> str:
        return str(self.sections.get(SECTION_DRAFT, "") or "")

    def get_critique(self) -> Optional[Critique]:
        return self.sections.get(SECTION_CRITIQUE)

    def get_final(self) -> str:
        return str(self.sections.get(SECTION_FINAL, "") or "")

    # --------------------------------------------------------------
    # Reporting
    # --------------------------------------------------------------

    def summary(self) -> str:
        lines: List[str] = []
        for sec in (SECTION_FACTS, SECTION_DRAFT, SECTION_CRITIQUE, SECTION_FINAL):
            v = self.sections.get(sec)
            if v is None:
                continue
            lines.append(f"  {sec}: {_summarise(v)}")
        return "\n".join(lines) or "  (scratchpad empty)"


def _summarise(value: Any) -> str:
    """One-line preview of a scratchpad value."""

    if isinstance(value, list):
        if not value:
            return "[]"
        first = value[0]
        if isinstance(first, Fact):
            return f"[{len(value)} facts; first: {first.text[:50]!r}]"
        return f"[{len(value)} items]"
    if isinstance(value, Critique):
        return f"verdict={value.verdict} issues={len(value.issues)}"
    text = str(value).replace("\n", " ").strip()
    return text[:90] + ("..." if len(text) > 90 else "")
