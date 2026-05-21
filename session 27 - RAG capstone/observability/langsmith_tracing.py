"""
Session 26 — observability/langsmith_tracing.py

LANGSMITH — hosted observability + eval platform from the LangChain team.

WHY THIS FILE
-------------
LangSmith gives every query in your pipeline a TRACE — one complete,
parent-child structured story of what happened. The S25 eval said
"faithfulness = 0.71"; the trace tells you WHICH chunk the model used,
WHICH retriever stage was slowest, WHICH judge call cost the most.

LangSmith is THE platform named in 2025-2026 senior AI engineer job
postings. The SDK is framework-agnostic — the `@traceable` decorator
works on any Python function, no LangChain rewrite required. Token
counts and dollar cost come through the hosted UI automatically; we do
not write a CostMeter.

USAGE — the production pattern
------------------------------
    from observability.langsmith_tracing import track, wrap_llm_client

    @track
    def answer(self, question: str) -> PipelineResult:
        ...

    client = wrap_llm_client(Anthropic())  # auto-instrumented

That's it. The decorator + the SDK wrapper ARE the integration. The
function body does NOT change. When `langsmith` is installed and
`LANGCHAIN_API_KEY` is set, every call produces a real LangSmith run
visible at https://smith.langchain.com.

OFFLINE FALLBACK
----------------
When LangSmith isn't reachable (a fresh student laptop, no API key, a
CI runner without secrets, an offline plane), the fallback appends to
.langsmith_traces.jsonl with a shape that mirrors LangSmith's `Run`
object. The demo, the debugger, and the cost-budget gate all read
either source uniformly via `get_offline_tracer()`.

The fallback exists so the lesson works without a LangSmith account.
The PRODUCTION pattern is unchanged: the decorator + the read API.

PRODUCTION PATTERNS INTRODUCED (S26):
  - decorator-based instrumentation             (NEW, named today)
  - auto-instrumentation of LLM SDKs            (NEW, named today)
  - graceful degradation when tooling isn't
    available                                   (NEW, named today)
  - use the platform, not custom code           (NEW, named today)
"""

from __future__ import annotations

import functools
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRACE_FILE = os.path.join(_HERE, "..", ".langsmith_traces.jsonl")
DEFAULT_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "session-26-rag")


# ── Detect LangSmith availability ───────────────────────────────────────────

try:
    from langsmith import traceable as _ls_traceable  # type: ignore
    from langsmith import Client as _LSClient          # type: ignore
    _HAS_LANGSMITH_PKG = True
except ImportError:
    _ls_traceable = None
    _LSClient = None
    _HAS_LANGSMITH_PKG = False

# LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY = real LangSmith path.
# We deliberately read the env at *call time* of `is_langsmith_active()`
# so a student can `os.environ[...] = ...` in a notebook and re-import.

def is_langsmith_active() -> bool:
    """True iff the `langsmith` package is installed AND the API key is set
    AND tracing is enabled. Reads env each call so notebooks can toggle."""
    if not _HAS_LANGSMITH_PKG:
        return False
    if not os.environ.get("LANGCHAIN_API_KEY"):
        return False
    if os.environ.get("LANGCHAIN_TRACING_V2", "true").lower() in (
            "false", "0", "no"):
        return False
    return True


# ── Auto-instrumentation: wrap the Anthropic / OpenAI SDK ───────────────────

def wrap_llm_client(client: Any) -> Any:
    """Auto-instrument an Anthropic or OpenAI SDK client.

    Returns the wrapped client. When LangSmith isn't active, returns the
    client unchanged — same call sites, no error.

    THE PRODUCTION PATTERN: one line. Token counts + cost come through
    automatically once wrapped. You never compute pricing manually.
    """
    if not is_langsmith_active():
        return client

    # Try Anthropic first, then OpenAI. Each wrapper is exposed by the
    # `langsmith.wrappers` namespace from langsmith>=0.1.0.
    try:
        from langsmith.wrappers import wrap_anthropic       # type: ignore
        # Heuristic: anthropic clients have a `.messages.create`.
        if hasattr(client, "messages") and hasattr(
                client.messages, "create"):
            return wrap_anthropic(client)
    except ImportError:
        pass

    try:
        from langsmith.wrappers import wrap_openai          # type: ignore
        if hasattr(client, "chat") and hasattr(
                client.chat, "completions"):
            return wrap_openai(client)
    except ImportError:
        pass

    # Unrecognised client — return as-is so caller still works.
    print(f"[langsmith] WARN: no wrapper for client type "
          f"{type(client).__name__}; tracing will miss LLM cost data.")
    return client


# ── Offline-fallback data shapes ────────────────────────────────────────────

@dataclass
class FallbackRun:
    """One run in the offline-fallback tracer.

    Mirrors the relevant fields of a LangSmith `Run` so the budget gate
    + debugger + downstream tools consume either source uniformly.
    """
    run_id: str
    trace_id: str            # root run_id for nested calls
    parent_run_id: Optional[str]
    name: str
    project: str
    start_time: float
    end_time: float
    duration_ms: float
    inputs: Dict[str, Any]
    outputs: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    # Token + cost bookkeeping. In the offline fallback we cannot pull
    # provider usage, so these are best-effort estimates the decorated
    # function can set via `set_current_run_metrics()`.
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


# ── The offline tracer (process-wide singleton) ─────────────────────────────

class _OfflineTracer:
    """Records traces to a JSONL file when LangSmith isn't reachable.

    NOT thread-safe by design — fellowship demos are single-threaded.
    A production deployment uses LangSmith directly, not this fallback,
    so the simplicity here is deliberate.
    """

    def __init__(self, trace_file: str = DEFAULT_TRACE_FILE) -> None:
        self.trace_file = os.path.abspath(trace_file)
        os.makedirs(os.path.dirname(self.trace_file), exist_ok=True)
        self._stack: List[FallbackRun] = []

    # ── Write API ─────────────────────────────────────────────────────────

    def start_run(self, name: str, inputs: Dict[str, Any]) -> FallbackRun:
        now = time.time()
        is_root = not self._stack
        parent = self._stack[-1] if self._stack else None
        run_id = uuid.uuid4().hex
        trace_id = run_id if is_root else parent.trace_id  # type: ignore[union-attr]
        run = FallbackRun(
            run_id=run_id,
            trace_id=trace_id,
            parent_run_id=(parent.run_id if parent else None),
            name=name,
            project=DEFAULT_PROJECT,
            start_time=now,
            end_time=now,
            duration_ms=0.0,
            inputs=_safe_jsonable(inputs),
            outputs=None,
        )
        self._stack.append(run)
        return run

    def finish_run(self, run: FallbackRun, output: Any,
                   error: Optional[str] = None,
                   **metric_updates: Any) -> None:
        now = time.time()
        run.end_time = now
        run.duration_ms = round((now - run.start_time) * 1000.0, 2)
        run.outputs = _safe_jsonable(output)
        if error:
            run.error = error
        for k, v in metric_updates.items():
            if hasattr(run, k):
                setattr(run, k, v)
            else:
                run.metadata[k] = v
        # Persist on EVERY run, not just root — so child runs are
        # individually inspectable (mirrors LangSmith's storage model).
        self._append_jsonl(run)
        # Pop only if we're unwinding the innermost open run.
        if self._stack and self._stack[-1].run_id == run.run_id:
            self._stack.pop()
        if not self._stack:
            print(f"[langsmith-offline] trace {run.trace_id[:8]}... persisted  "
                  f"({run.duration_ms:.0f}ms, "
                  f"${run.cost_usd:.4f}, {run.total_tokens}tok)")

    def _append_jsonl(self, run: FallbackRun) -> None:
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(run), default=str) + "\n")
        except OSError as e:
            print(f"[langsmith-offline] WARN: persist failed ({e!s})")

    # ── Read API ──────────────────────────────────────────────────────────

    def get_recent(self, n: int = 30,
                   only_roots: bool = True) -> List[Dict[str, Any]]:
        """Return the last N persisted runs, newest first.

        only_roots=True returns one entry per trace (the root run). Set
        False to see every nested run.
        """
        runs = self._read_all()
        if only_roots:
            runs = [r for r in runs if r.get("parent_run_id") is None]
        runs.sort(key=lambda r: r.get("start_time", 0.0), reverse=True)
        return runs[:n]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        for r in self._read_all():
            if r.get("run_id") == run_id or r.get("trace_id") == run_id:
                return r
        return None

    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Return ALL runs (root + children) belonging to one trace."""
        return [r for r in self._read_all()
                if r.get("trace_id") == trace_id]

    def _read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.trace_file):
            return []
        out: List[Dict[str, Any]] = []
        with open(self.trace_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    # ── Current-run helper (for `set_current_run_metrics`) ────────────────

    def current_run(self) -> Optional[FallbackRun]:
        return self._stack[-1] if self._stack else None


# Process-singleton (lazy-init).
_offline_singleton: Optional[_OfflineTracer] = None


def get_offline_tracer() -> _OfflineTracer:
    """Return the process-wide offline tracer.

    In a LangSmith-enabled environment, prefer querying LangSmith's SDK
    directly for trace reads; this is the offline path. The demo +
    debugger + budget gate use this so the lesson works without a
    LangSmith account.
    """
    global _offline_singleton
    if _offline_singleton is None:
        _offline_singleton = _OfflineTracer()
    return _offline_singleton


# ── The unified `track` decorator ───────────────────────────────────────────

def track(fn: Optional[Callable] = None, *,
          name: Optional[str] = None) -> Callable:
    """Decorator that turns a function into a traced run.

    Behaviour:
      - If LangSmith is installed AND configured (LANGCHAIN_API_KEY set
        and LANGCHAIN_TRACING_V2 not false), defers to
        `langsmith.traceable` — real runs appear in the hosted UI.
      - Otherwise, uses the offline fallback that appends to a JSONL
        file under .langsmith_traces.jsonl.

    Either way, the FUNCTION BODY DOES NOT CHANGE. That's the whole
    point of the pattern.
    """
    def decorator(func: Callable) -> Callable:
        if is_langsmith_active() and _ls_traceable is not None:
            return _ls_traceable(name=name or func.__name__)(func)

        # Fallback path.
        run_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_offline_tracer()
            input_summary = {
                "args": _summarise_args(args[1:] if args else ()),
                "kwargs": _safe_jsonable(kwargs),
            }
            run = tracer.start_run(run_name, input_summary)
            try:
                result = func(*args, **kwargs)
                tracer.finish_run(run, output=result)
                return result
            except Exception as e:
                tracer.finish_run(run, output=None, error=repr(e))
                raise
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


# ── Convenience: explicit context-manager runs ──────────────────────────────

@contextmanager
def trace_span(name: str, **inputs: Any):
    """Open a manually-named run for a block of code that isn't a function.

    Use this when you can't decorate (e.g. you need a sub-section of one
    function to be its own run). Updates apply to the offline tracer;
    inside a real LangSmith environment, decorators are the idiom.

    Usage:
        with trace_span("retrieve", query=q, k=4) as r:
            ...
            r.cost_usd = 0.0023
            r.input_tokens = 1240
    """
    tracer = get_offline_tracer()
    run = tracer.start_run(name, inputs)
    try:
        yield run
        tracer.finish_run(run, output=None)
    except Exception as e:
        tracer.finish_run(run, output=None, error=repr(e))
        raise


def set_current_run_metrics(*,
                            input_tokens: Optional[int] = None,
                            output_tokens: Optional[int] = None,
                            cost_usd: Optional[float] = None) -> None:
    """Attach token / cost metrics to the run currently being recorded.

    Only meaningful in the offline-fallback path (LangSmith pulls these
    from the LLM response automatically when `wrap_llm_client()` is used).
    No-op when no run is active.
    """
    tracer = get_offline_tracer()
    run = tracer.current_run()
    if run is None:
        return
    if input_tokens is not None:
        run.input_tokens = int(input_tokens)
    if output_tokens is not None:
        run.output_tokens = int(output_tokens)
    if input_tokens is not None or output_tokens is not None:
        run.total_tokens = run.input_tokens + run.output_tokens
    if cost_usd is not None:
        run.cost_usd = float(cost_usd)


# ── LangSmith UI deep-link ──────────────────────────────────────────────────

def get_project_url() -> Optional[str]:
    """Return the URL to the current LangSmith project, or None offline."""
    if not is_langsmith_active():
        return None
    project = os.environ.get("LANGCHAIN_PROJECT", DEFAULT_PROJECT)
    # The hosted endpoint may be overridden by self-host deploys.
    endpoint = os.environ.get(
        "LANGCHAIN_ENDPOINT", "https://smith.langchain.com")
    return f"{endpoint}/o/-/projects/p/{project}"


def get_run_url(run_id: str) -> Optional[str]:
    """Return the LangSmith UI URL for a given run_id, or None offline."""
    if not is_langsmith_active():
        return None
    endpoint = os.environ.get(
        "LANGCHAIN_ENDPOINT", "https://smith.langchain.com")
    project = os.environ.get("LANGCHAIN_PROJECT", DEFAULT_PROJECT)
    return f"{endpoint}/o/-/projects/p/{project}/r/{run_id}"


# ── Internal helpers ────────────────────────────────────────────────────────

def _safe_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Make `obj` JSON-serialisable, truncating large strings/lists."""
    if _depth > 4:
        return repr(obj)[:200]
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) < 500 else obj[:500] + "..."
    if isinstance(obj, (list, tuple)):
        return [_safe_jsonable(x, _depth + 1) for x in obj[:20]]
    if isinstance(obj, dict):
        return {str(k): _safe_jsonable(v, _depth + 1)
                for k, v in list(obj.items())[:20]}
    if hasattr(obj, "__dict__"):
        return _safe_jsonable(vars(obj), _depth + 1)
    return repr(obj)[:200]


def _summarise_args(args: tuple) -> List[Any]:
    return [_safe_jsonable(a) for a in args]


# ── A demo when the file is run directly ────────────────────────────────────

if __name__ == "__main__":
    print(f"[langsmith] LANGSMITH_ACTIVE = {is_langsmith_active()}")
    print(f"[langsmith] offline file     = {get_offline_tracer().trace_file}")
    if is_langsmith_active():
        print(f"[langsmith] project URL       = {get_project_url()}")
    print()

    @track
    def retrieve(query: str, k: int = 3) -> List[str]:
        time.sleep(0.05)
        return [f"chunk{i}" for i in range(k)]

    @track
    def generate(question: str, context: List[str]) -> str:
        time.sleep(0.15)
        set_current_run_metrics(input_tokens=1200, output_tokens=240,
                                 cost_usd=0.0034)
        return f"answer using {len(context)} chunks"

    @track
    def answer(question: str) -> str:
        ctx = retrieve(question, k=4)
        return generate(question, ctx)

    answer("How does LangSmith tracing work?")
    print()
    recent = get_offline_tracer().get_recent(n=3, only_roots=True)
    for r in recent:
        print(f"  trace {r['trace_id'][:8]}...  {r['name']:<24}  "
              f"{r['duration_ms']:.0f}ms  ${r.get('cost_usd', 0):.4f}")
