"""
Session 37 — demo.py

Five walkthrough demos for the combined LangChain session.

    python demo.py 1   # Pure Python vs LangChain — same task, two code shapes
    python demo.py 2   # LCEL: prompt | model | parser + variable substitution
    python demo.py 3   # Document Q&A chain (retrieve -> stuff -> answer)
    python demo.py 4   # Summarisation chain with streaming
    python demo.py 5   # Callbacks (observability), caching, and debugging
    python demo.py all  # run every demo

Real Anthropic calls by default (needs ANTHROPIC_API_KEY + langchain-anthropic).
Offline, deterministic, no key:

    PYTHONPYCACHEPREFIX=/tmp/s37_pycache FAKE_LLM=1 python demo.py all
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


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

# Hard-disable LangChain's automatic LangSmith / hosted tracing.
# This session teaches observability via a hand-built callback handler;
# we never want background calls to an external tracing service,
# regardless of any LANGCHAIN_TRACING_V2 / LANGSMITH_API_KEY the user has
# set in their shell or .env. Force it off before any chain runs.
for _flag in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING", "LANGSMITH_TRACING"):
    os.environ[_flag] = "false"
for _key in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY",
             "LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT"):
    os.environ.pop(_key, None)


if os.environ.get("FAKE_LLM", "") != "1" and not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit(
        "\nERROR: ANTHROPIC_API_KEY is not set.\n"
        "  1) export ANTHROPIC_API_KEY=sk-ant-...   (real calls)\n"
        "  2) export FAKE_LLM=1                      (offline mode)\n"
    )


# Imports after env is loaded so FAKE_LLM is visible to model_factory.
from callbacks import StepLoggingCallbackHandler  # noqa: E402
from chains import (  # noqa: E402
    build_qa_chain,
    build_summary_chain,
    build_template_chain,
)
from corpus import corpus_as_one_blob  # noqa: E402
from model_factory import get_chat_model, is_fake  # noqa: E402
from pure_python_baseline import summarise_pure_python  # noqa: E402
from trace_logger import (  # noqa: E402
    c_amber,
    c_dim,
    c_mint,
    print_final_answer,
    print_note,
    print_section,
    print_subheader,
    print_user_question,
)


def _mode_banner() -> None:
    mode = "FAKE_LLM (offline, canned)" if is_fake() else "REAL Anthropic API"
    print(c_dim(f"  [mode: {mode}]"))


# ---------------------------------------------------------------------
# Demo 1 — Pure Python vs LangChain
# ---------------------------------------------------------------------


def demo_1() -> None:
    print_section("Demo 1 — Pure Python vs LangChain (same task)")
    print("  Task: summarise the NovaDesk corpus. Same job, two code shapes.")
    _mode_banner()
    document = corpus_as_one_blob()

    print_subheader("Pure Python (S29/S30 style) — you wire every step")
    print_note("build messages -> call SDK -> dig text out of response (3 steps)")
    pp_answer = summarise_pure_python(document)
    print_final_answer(pp_answer)

    print_subheader("LangChain LCEL — three links, one line")
    print_note("chain = prompt | model | parser   ;   chain.invoke({...})")
    from prompts import SUMMARISER_SYSTEM
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages(
        [("system", SUMMARISER_SYSTEM), ("human", "Summarise:\n\n{document}")]
    )
    chain = prompt | get_chat_model() | StrOutputParser()
    lc_answer = chain.invoke({"document": document})
    print_final_answer(lc_answer)

    print_subheader("What LangChain did for free")
    print_note("message assembly, the API call, and text extraction — all hidden.")
    print_note("What it HIDES: the exact request, retries, and token usage. You")
    print_note("trade visibility for brevity. Use it when that trade is worth it.")


# ---------------------------------------------------------------------
# Demo 2 — LCEL: prompt | model | parser
# ---------------------------------------------------------------------


def demo_2() -> None:
    print_section("Demo 2 — LCEL: prompt | model | parser")
    print("  The smallest possible chain. Watch the {topic} placeholder fill in.")
    _mode_banner()
    chain = build_template_chain()

    for topic in ["LangChain", "an output parser"]:
        print_user_question(f"topic = {topic!r}")
        answer = chain.invoke({"topic": topic})
        print_final_answer(answer)

    print_subheader("Why this matters")
    print_note("The same chain handled two different inputs. The template is the")
    print_note("reusable part; invoke() supplies the variable. This is LCEL.")


# ---------------------------------------------------------------------
# Demo 3 — Document Q&A chain
# ---------------------------------------------------------------------


def demo_3() -> None:
    print_section("Demo 3 — Document Q&A chain (retrieve -> stuff -> answer)")
    print("  One LCEL chain does retrieval-augmented Q&A over the NovaDesk docs.")
    _mode_banner()
    qa = build_qa_chain(k=2)

    questions = [
        "What is the refund policy?",
        "What uptime does the Pro plan guarantee?",
        "Does NovaDesk integrate with Slack?",
    ]
    for q in questions:
        print_user_question(q)
        answer = qa.invoke(q)
        print_final_answer(answer)


# ---------------------------------------------------------------------
# Demo 4 — Summarisation chain with streaming
# ---------------------------------------------------------------------


def demo_4() -> None:
    print_section("Demo 4 — Summarisation chain WITH streaming")
    print("  Same three-link chain, but .stream() yields tokens as they arrive.")
    _mode_banner()
    chain = build_summary_chain()
    document = corpus_as_one_blob()

    print_user_question("Summarise the NovaDesk corpus (streamed)")
    print(f"\n{c_mint('ANSWER (streaming):')}")
    print("  ", end="", flush=True)
    for token in chain.stream({"document": document}):
        print(token, end="", flush=True)
        time.sleep(0.01)  # make the streaming visible to the eye
    print()
    print_subheader("Why stream?")
    print_note("On a long answer the user sees the first words in ~50 ms instead")
    print_note("of waiting for the whole thing. invoke() -> stream() is one word.")


# ---------------------------------------------------------------------
# Demo 5 — Callbacks (observability), caching, and debugging
# ---------------------------------------------------------------------


def demo_5() -> None:
    print_section("Demo 5 — Callbacks, caching, and debugging a wrong answer")
    _mode_banner()

    # -- 5a: callbacks log every step --------------------------------------
    print_subheader("5a — Callback-based observability (PROD PATTERN)")
    print_note("Pass a handler at call time; LangChain calls it on every step.")
    qa = build_qa_chain(k=2)
    handler = StepLoggingCallbackHandler()
    q = "What is the refund policy?"
    print_user_question(q)
    answer = qa.invoke(q, config={"callbacks": [handler]})
    print_final_answer(answer)
    print_note(f"handler saw {handler.llm_calls} model call(s).")

    # -- 5b: caching cuts a repeat call ------------------------------------
    print_subheader("5b — Response caching (PROD PATTERN)")
    from langchain_core.caches import InMemoryCache
    from langchain_core.globals import set_llm_cache

    set_llm_cache(InMemoryCache())
    model = get_chat_model()
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from prompts import TEMPLATE_DEMO_SYSTEM

    cached_chain = (
        ChatPromptTemplate.from_messages(
            [("system", TEMPLATE_DEMO_SYSTEM), ("human", "Explain: {topic}")]
        )
        | model
        | StrOutputParser()
    )
    payload = {"topic": "caching"}

    before = getattr(model, "call_count", None)
    t0 = time.time()
    cached_chain.invoke(payload)
    first_ms = (time.time() - t0) * 1000
    t1 = time.time()
    cached_chain.invoke(payload)  # identical input -> should hit the cache
    second_ms = (time.time() - t1) * 1000
    after = getattr(model, "call_count", None)

    print_note(f"first call:  {first_ms:.0f} ms")
    print_note(f"second call: {second_ms:.0f} ms (identical input)")
    if before is not None and after is not None:
        print_note(
            f"model._generate fired {after - before} time(s) for 2 invocations "
            "— the 2nd was served from cache."
        )
    set_llm_cache(None)  # reset global cache so other demos are unaffected

    # -- 5c: debugging a wrong answer --------------------------------------
    print_subheader("5c — Debugging a wrong answer with the trace")
    print_note("Ask something NOT in the docs. A naive chain would hallucinate.")
    bad_q = "What is the NovaDesk share price?"
    handler2 = StepLoggingCallbackHandler()
    print_user_question(bad_q)
    bad_answer = qa.invoke(bad_q, config={"callbacks": [handler2]})
    print_final_answer(bad_answer)
    print_note("The trace shows the retriever found no matching source, so the")
    print_note("model correctly refused. Reading the trace IS the debugging.")


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
    if key == "all":
        for fn in _DEMOS.values():
            fn()
        return 0
    fn = _DEMOS.get(key)
    if fn is None:
        print(f"Unknown demo '{key}'. Try one of: {', '.join(_DEMOS)}, all")
        return 2
    fn()
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
