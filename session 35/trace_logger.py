"""
Session 35 — trace_logger.py

ANSI-coloured trace printer for the hierarchical + debate + competitive
demos. Same shape as S34's trace logger, with new event types added
for hierarchical layers and debate transcripts.

ENVIRONMENT:
  - NO_COLOR=1 disables ANSI codes (useful when piping to a file).
"""

from __future__ import annotations

import os
from typing import Any, Dict


_USE_COLOR = os.environ.get("NO_COLOR", "") != "1"


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def red(t: str) -> str:    return _c("31", t)
def green(t: str) -> str:  return _c("32", t)
def yellow(t: str) -> str: return _c("33", t)
def blue(t: str) -> str:   return _c("34", t)
def magenta(t: str) -> str:return _c("35", t)
def cyan(t: str) -> str:   return _c("36", t)
def grey(t: str) -> str:   return _c("90", t)
def bold(t: str) -> str:   return _c("1", t)


def banner(text: str) -> None:
    bar = "=" * max(60, len(text) + 4)
    print()
    print(red(bar))
    print(red(f"  {text}"))
    print(red(bar))


def section(text: str) -> None:
    print()
    print(yellow(f"---- {text} ----"))


def kv(label: str, value: Any) -> None:
    print(f"  {bold(label)}: {value}")


def handle_event(evt: Dict[str, Any]) -> None:
    """Pretty-print a typed trace event. Every code module that wants to
    surface progress calls this with a small dict."""

    t = evt.get("type", "event")

    if t == "director_decision":
        next_mgr = evt.get("next_manager", "?")
        done = evt.get("done", False)
        if done:
            print(magenta(f"  [director] DONE — publishing final answer"))
        else:
            print(magenta(f"  [director] -> {next_mgr}: ")
                  + evt.get("instruction", ""))

    elif t == "manager_action":
        mgr = evt.get("manager", "?")
        action = evt.get("action", "?")
        target = evt.get("target", "?")
        print(blue(f"  [{mgr}] {action} -> {target}: ")
              + evt.get("instruction", ""))

    elif t == "subtopic_split":
        mgr = evt.get("manager", "?")
        topics = evt.get("sub_topics", [])
        print(blue(f"  [{mgr}] split into {len(topics)} sub-topics:"))
        for i, topic in enumerate(topics, 1):
            print(f"    {i}. {topic}")

    elif t == "researcher_finished":
        rid = evt.get("researcher_id", "?")
        n = evt.get("num_facts", 0)
        print(green(f"  [researcher:{rid}] returned {n} facts"))

    elif t == "merge_done":
        n = evt.get("num_facts", 0)
        print(green(f"  [research_manager] merged into {n} total facts"))

    elif t == "writer_finished":
        rev = evt.get("revision", False)
        words = evt.get("words", 0)
        tag = " (revision)" if rev else ""
        print(green(f"  [writer] returned {words} word draft{tag}"))

    elif t == "fact_check_verdict":
        verdict = evt.get("verdict", "?")
        issues = evt.get("issues", [])
        if verdict == "ACCEPT":
            print(green(f"  [fact_checker] ACCEPT"))
        else:
            print(red(f"  [fact_checker] REVISE — {len(issues)} issues"))
            for issue in issues[:3]:
                print(red(f"    - {issue}"))

    elif t == "editorial_report":
        status = evt.get("status", "?")
        if status == "ACCEPTED":
            print(magenta(f"  [editorial_manager] REPORT -> director: ACCEPTED"))
        elif status == "ESCALATED":
            print(red(f"  [editorial_manager] ESCALATE -> director: "
                     "max revisions hit"))

    elif t == "termination":
        reason = evt.get("reason", "?")
        rounds = evt.get("rounds", 0)
        calls = evt.get("calls", 0)
        print(yellow(f"  >>> TERMINATED — reason={reason} "
                     f"rounds={rounds} llm_calls={calls}"))

    # --- debate panel events ---------------------------------------------

    elif t == "debate_round_start":
        n = evt.get("round_num", 0)
        print(cyan(f"  [debate] round {n} starting"))

    elif t == "panelist_argument":
        panelist = evt.get("panelist", "?")
        claim = evt.get("claim", "")
        if "bull" in panelist:
            print(green(f"  [{panelist}] {claim}"))
        elif "bear" in panelist:
            print(red(f"  [{panelist}] {claim}"))
        else:
            print(grey(f"  [{panelist}] {claim}"))

    elif t == "consensus":
        report = evt.get("report", {})
        agreed = report.get("agreed_points", [])
        disagree = report.get("disagreements", [])
        conf = report.get("confidence", 0.0)
        print(magenta(f"  [moderator] consensus — "
                      f"{len(agreed)} agreed, {len(disagree)} disagreements, "
                      f"confidence={conf:.2f}"))

    # --- competitive panel events ----------------------------------------

    elif t == "candidate_drafted":
        cid = evt.get("candidate_id", "?")
        style = evt.get("style", "?")
        words = evt.get("words", 0)
        print(blue(f"  [candidate:{cid}] style={style} {words}w"))

    elif t == "judge_verdict":
        winner = evt.get("winner_id", "?")
        scores = evt.get("scores", {})
        print(magenta(f"  [judge] WINNER = {winner}"))
        for cid, s in scores.items():
            tag = " <-- winner" if cid == winner else ""
            print(grey(f"    {cid}: acc={s.get('accuracy')} "
                      f"cla={s.get('clarity')} "
                      f"use={s.get('usefulness')}{tag}"))

    # --- message bus / communication protocol events ---------------------

    elif t == "message_sent":
        sender = evt.get("sender", "?")
        recipient = evt.get("recipient", "?")
        intent = evt.get("intent", "?")
        subject = evt.get("subject", "")
        print(grey(f"  [bus] {sender} -> {recipient} [{intent}] {subject}"))

    elif t == "bus_validation":
        problems = evt.get("problems", [])
        if not problems:
            print(green("  [bus] validation: clean"))
        else:
            print(red(f"  [bus] validation: {len(problems)} problems"))
            for p in problems[:5]:
                print(red(f"    - {p}"))

    else:
        # Fallback printer.
        print(grey(f"  [{t}] {evt}"))
