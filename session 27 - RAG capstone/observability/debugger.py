"""
Session 26 -- observability/debugger.py

THE PATTERN -- replay a failed query side-by-side with an alternative
configuration. The original query lives in LangSmith (pulled by run_id);
the replay runs locally and produces a new trace.

WHY THIS FILE
-------------
The S25 diagnostic told the engineer WHICH LAYER to fix. The trace
told them WHICH SPAN went wrong. This file gives them the third move:
REPLAY THE QUERY with a different config and compare the new trace to
the old trace.

Two replay knobs out of the box:
  1. Swap the answer model (e.g. Haiku -> Sonnet)
  2. Swap top-K (e.g. k=4 -> k=8)

Both produce a second trace; the replay function prints a diff so the
engineer can decide which fix to commit.

In a LangSmith-enabled environment, both traces appear in the hosted
UI side-by-side. In the offline fallback, both runs end up in
.langsmith_traces.jsonl and the printed diff covers latency, cost, and
answer.

PRODUCTION PATTERNS INTRODUCED (S26):
  - trace replay with one knob at a time         (NEW, named today)
  - side-by-side trace diff                       (NEW, named today)
  - replay-driven fix proposal                    (NEW, named today)
  - pull-from-platform-by-run-id                  (NEW, named today)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .langsmith_tracing import (
    is_langsmith_active,
    get_offline_tracer,
    get_run_url,
    DEFAULT_PROJECT,
)


@dataclass
class ReplayResult:
    """The output of one replay run."""
    question: str
    original_run: Optional[Dict[str, Any]]
    new_run: Optional[Dict[str, Any]]
    knob: str
    knob_value: str
    answer_diff: str
    original_url: Optional[str]
    new_url: Optional[str]


def fetch_original_run(run_id: str,
                       *,
                       project: str = DEFAULT_PROJECT) -> Optional[Dict[str, Any]]:
    """Pull the original run by ID. Tries LangSmith first, falls back to
    the offline JSONL log."""
    if is_langsmith_active():
        try:
            from langsmith import Client                  # type: ignore
            client = Client()
            run = client.read_run(run_id=run_id)
            return {
                "run_id":       str(run.id),
                "trace_id":     str(getattr(run, "trace_id", run.id)),
                "name":         run.name,
                "duration_ms":  _ms_between(run.start_time, run.end_time),
                "inputs":       run.inputs,
                "outputs":      run.outputs,
                "cost_usd":     float(getattr(run, "total_cost", 0.0) or 0.0),
                "total_tokens": int(getattr(run, "total_tokens", 0) or 0),
                "source":       "langsmith",
            }
        except Exception as e:
            print("[debugger] WARN: LangSmith fetch failed (" + str(e)
                  + ") -- trying offline log.")

    run = get_offline_tracer().get_run(run_id)
    if run is None:
        return None
    run["source"] = "offline"
    return run


def replay(pipeline_factory: Callable[..., Any],
           *,
           run_id: Optional[str] = None,
           question: Optional[str] = None,
           knob: str = "model",
           new_value: str = "claude-sonnet-4-6",
           verbose: bool = True) -> ReplayResult:
    """Re-run a failed query with one knob changed.

    Args:
        pipeline_factory: callable returning a fresh pipeline.
                          Signature: factory(model=None, k=None, **kw)
                          -> RAGPipeline.
        run_id:           the LangSmith (or offline) run_id of the
                          original query. If absent, `question` must be
                          provided.
        question:         the question to replay. If absent, pulled
                          from the original run's inputs.
        knob:             "model" or "k" (top-K) for now.
        new_value:        new value for that knob.
    """
    original_run: Optional[Dict[str, Any]] = None
    original_answer = "(no original run available)"
    original_url: Optional[str] = None

    if run_id is not None:
        original_run = fetch_original_run(run_id)
        if original_run is None and verbose:
            print("[debugger] WARN: no run found for id=" + run_id[:8] + "...")
        else:
            original_url = get_run_url(run_id)
            original_answer = _extract_answer(original_run) or original_answer
            if question is None:
                question = _extract_question(original_run)

    if question is None:
        raise ValueError(
            "replay() needs either a run_id or an explicit question.")

    # Build replay pipeline with the one knob changed.
    if knob == "model":
        pipeline = pipeline_factory(model=new_value)
    elif knob == "k":
        pipeline = pipeline_factory(k=int(new_value))
    else:
        pipeline = pipeline_factory(**{knob: new_value})

    if verbose:
        print("[debugger] replaying  question=" + repr(question)
              + "  knob=" + knob + "=" + str(new_value))
    result = pipeline.answer(question)

    # Grab the new run from the offline tracer (the most recent root).
    new_run = None
    recent = get_offline_tracer().get_recent(n=1, only_roots=True)
    if recent:
        new_run = recent[0]

    new_url = None
    if new_run and is_langsmith_active():
        new_url = get_run_url(new_run["run_id"])

    new_answer = getattr(result, "answer", "") or "(no answer)"
    answer_diff = _short_diff(original_answer, new_answer)

    if verbose:
        _print_diff(original_run, new_run, knob, new_value,
                    original_answer, new_answer,
                    original_url, new_url)

    return ReplayResult(
        question=question,
        original_run=original_run,
        new_run=new_run,
        knob=knob,
        knob_value=str(new_value),
        answer_diff=answer_diff,
        original_url=original_url,
        new_url=new_url,
    )


# Diff helpers.

def _extract_answer(run: Optional[Dict[str, Any]]) -> str:
    """Pull a likely answer field out of a run's outputs."""
    if not run:
        return ""
    out = run.get("outputs")
    if isinstance(out, dict):
        # The pipeline returns a PipelineResult; dict-ified it has 'answer'.
        for key in ("answer", "text", "completion", "output"):
            if key in out:
                return str(out[key])
        return str(out)
    if isinstance(out, str):
        return out
    return str(out) if out is not None else ""


def _extract_question(run: Dict[str, Any]) -> str:
    """Best-effort extraction of the original question from run inputs."""
    inputs = run.get("inputs") or {}
    if isinstance(inputs, dict):
        kwargs = inputs.get("kwargs") or {}
        if isinstance(kwargs, dict) and "question" in kwargs:
            return str(kwargs["question"])
        args = inputs.get("args") or []
        if args and isinstance(args, list):
            return str(args[0])
        for key in ("question", "query", "input"):
            if key in inputs:
                return str(inputs[key])
    return ""


def _short_diff(a: str, b: str, max_len: int = 200) -> str:
    if a == b:
        return "(identical answers)"
    a_short = a[:max_len] + ("..." if len(a) > max_len else "")
    b_short = b[:max_len] + ("..." if len(b) > max_len else "")
    return "BEFORE: " + a_short + "\n----\nAFTER:  " + b_short


def _print_diff(original_run, new_run, knob, new_value,
                original_answer, new_answer,
                original_url, new_url):
    print()
    print("=" * 78)
    print("  TRACE REPLAY DIFF  -  knob=" + knob + "=" + str(new_value))
    print("=" * 78)

    def _summ(run):
        if not run:
            return "(n/a)"
        ms = run.get("duration_ms", 0)
        cost = run.get("cost_usd", 0) or 0
        return ("%.0fms, $%.4f" % (ms, cost))

    print("  ORIGINAL  run: " + _summ(original_run))
    if original_url:
        print("            UI: " + original_url)
    print("  REPLAY    run: " + _summ(new_run))
    if new_url:
        print("            UI: " + new_url)

    if original_run and new_run:
        d_ms = new_run.get("duration_ms", 0) - original_run.get("duration_ms", 0)
        d_cost = ((new_run.get("cost_usd", 0) or 0)
                  - (original_run.get("cost_usd", 0) or 0))
        print("  DELTA        : %+.0fms, $%+.4f" % (d_ms, d_cost))

    print()
    print("  ANSWER DIFF:")
    print("  " + _short_diff(original_answer, new_answer).replace("\n", "\n  "))
    print("=" * 78)


def _ms_between(t0, t1) -> float:
    try:
        return (t1 - t0).total_seconds() * 1000.0
    except Exception:
        return 0.0


if __name__ == "__main__":
    print("[debugger] this is a teaching utility -- see demo.py act 4.")
    print("[debugger] cannot run standalone without a pipeline factory.")
