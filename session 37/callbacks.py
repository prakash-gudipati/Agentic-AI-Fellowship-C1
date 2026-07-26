"""
Session 37 — callbacks.py

A callback handler that logs every step a chain takes. This is LangChain's
built-in observability hook: you pass a handler in at call time, and
LangChain calls your methods as the chain runs — chain starts, the model
is called, the model returns, the chain ends.

Analogy: a flight recorder. You don't change how the plane flies; you bolt
on a box that records every event so you can replay what happened when
something goes wrong. This is the same idea the S26 observability session taught —
here we do it by hand so students see what the platform automates.

PROD PATTERN — Callback-Based Observability.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from langchain_core.callbacks.base import BaseCallbackHandler

from trace_logger import c_amber, c_blue, c_dim, c_mint


class StepLoggingCallbackHandler(BaseCallbackHandler):
    """Prints a labelled line for each step of a chain run.

    Plug it in per call:  chain.invoke(x, config={"callbacks": [handler]})
    """

    # Generic wiring nodes LangChain emits internally. We skip them so the
    # trace shows only the steps worth teaching: the prompt and the model.
    _NOISY = {
        "chain", "RunnableSequence", "RunnableParallel",
        "RunnablePassthrough", "RunnableLambda", "RunnableAssign", "",
    }

    def __init__(self) -> None:
        self.llm_calls = 0
        self._llm_started_at = 0.0

    # -- chain-level events ------------------------------------------------

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Any, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or ""
        if name in self._NOISY:
            return
        print(c_blue(f"  [step]          {name}"))

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        return  # end markers are noise once we only show named steps

    # -- model-level events ------------------------------------------------

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.llm_calls += 1
        self._llm_started_at = time.time()
        preview = prompts[0].replace("\n", " ")[:70] if prompts else ""
        print(c_amber(f"  [llm start]     call #{self.llm_calls} - prompt: {preview}..."))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        elapsed_ms = (time.time() - self._llm_started_at) * 1000
        print(c_mint(f"  [llm end]       {elapsed_ms:.0f} ms"))

    # -- error event -------------------------------------------------------

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        print(c_amber(f"  [chain ERROR]   {type(error).__name__}: {error}"))
