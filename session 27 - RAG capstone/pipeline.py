"""
Session 26 — pipeline.py  (instrumented with LangSmith)

S26 carries over the S25 pipeline UNCHANGED IN BEHAVIOUR. The only
additions are:

  * @track on answer() — every call produces a traced LangSmith run
  * @track on _generate() — the inner LLM call is its own child run
  * wrap_llm_client(client) — auto-instruments the Anthropic SDK so
    token counts + cost flow through to LangSmith automatically

The pipeline body still does the same retrieve → prompt → generate.
That's the whole point: the production pattern is decorate, don't
rewrite. S20–S25 code does not change to gain observability.

Cost tracking is NOT done here — LangSmith captures it from the
provider response when `wrap_llm_client` is used. This is the
production rule:  USE THE PLATFORM, NOT CUSTOM CODE.

Production patterns reinforced (S26):
  - decorator-based instrumentation             (S26 new)
  - auto-instrumentation of SDK clients         (S26 new)
  - use the platform, not custom code           (S26 new)
  - structured return type (PipelineResult)     (S22)
  - dependency injection of the retriever       (S22)
  - try/except on every external API call       (S3)
  - env var for API key                         (S8)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from retrievers.base import Retrieved, Retriever
from observability.langsmith_tracing import (track, wrap_llm_client,
                                              trace_span,
                                              set_current_run_metrics)


load_dotenv()

# ── Defaults ────────────────────────────────────────────────────────────────
ANSWER_MODEL = "claude-haiku-4-5-20251001"
ANSWER_MAX_TOKENS = 600
DEFAULT_K = 4

SYSTEM_PROMPT = (
    "You answer questions strictly from the supplied context. "
    "Quote relevant sentences when useful, but do NOT invent facts. "
    "If the context does not contain the answer, reply exactly: "
    "'I don't know based on the provided context.' "
    "Keep the answer under five sentences."
)

USER_TEMPLATE = (
    "Context:\n{context_block}\n\n"
    "Question: {question}\n\n"
    "Answer using only the context above. Be specific and concise."
)


# ── Public return type ──────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """One end-to-end run of the pipeline.

    Note on cost fields: when LangSmith is active, cost lives in the
    LangSmith UI — you query it from there, not from this dataclass.
    The fields stay on the dataclass so the S25 tests + offline fallback
    keep working uniformly. Set automatically when offline; left at 0.0
    when LangSmith owns the cost data.
    """
    question:        str
    answer:          str
    contexts:        List[str]
    retrieved:       List[Retrieved]
    # offline-fallback bookkeeping (LangSmith owns these in prod)
    cost_usd:        float = 0.0
    input_tokens:    int   = 0
    output_tokens:   int   = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question":          self.question,
            "answer":            self.answer,
            "contexts":          self.contexts,
            "retrieved_sources": [r.chunk.source for r in self.retrieved],
            "retrieved_scores":  [round(r.score, 4) for r in self.retrieved],
            "cost_usd":          round(self.cost_usd, 6),
            "input_tokens":      self.input_tokens,
            "output_tokens":     self.output_tokens,
        }


# ── Lightweight offline pricing (for the fallback only) ─────────────────────
# In real production, this table doesn't exist here — LangSmith owns it.
# This snippet is ONLY used when the demo runs without LANGCHAIN_API_KEY,
# so the offline JSONL still shows a non-zero cost number.

_OFFLINE_PRICE_PER_MTOK = {
    # USD per million tokens — input, output
    "claude-haiku-4-5-20251001":  (1.00, 5.00),
    "claude-sonnet-4-6":          (3.00, 15.00),
    "claude-opus-4-6":            (15.00, 75.00),
}


def _offline_cost(model: str, in_tok: int, out_tok: int) -> float:
    price_in, price_out = _OFFLINE_PRICE_PER_MTOK.get(model, (1.0, 5.0))
    return (in_tok / 1_000_000.0) * price_in + \
           (out_tok / 1_000_000.0) * price_out


# ── The pipeline ────────────────────────────────────────────────────────────

class RAGPipeline:
    """Question → retrieved chunks → LLM answer, with observability built in."""

    def __init__(self,
                 retriever: Retriever,
                 *,
                 k: int = DEFAULT_K,
                 model: str = ANSWER_MODEL,
                 system_prompt: str = SYSTEM_PROMPT,
                 client: Any = None) -> None:
        self.retriever = retriever
        self.k = k
        self.model = model
        self.system_prompt = system_prompt
        self._client = client

    # ── External resources (lazy) ──────────────────────────────────────────

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError("anthropic package not installed.") from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env "
                "and add your key."
            )
        # Auto-instrument the client. When LangSmith is active, every
        # `client.messages.create(...)` call becomes a child run with
        # token counts + dollar cost. When LangSmith isn't active, this
        # returns the client unchanged.
        self._client = wrap_llm_client(Anthropic(api_key=api_key))
        return self._client

    # ── Main entry point ───────────────────────────────────────────────────

    @track(name="RAGPipeline.answer")
    def answer(self, question: str) -> PipelineResult:
        """Retrieve, prompt, generate. Produces ONE LangSmith trace per call."""

        # ── STAGE 1 — retrieve ──────────────────────────────────────────
        print(f"\n[pipeline] step 1/3 — retrieve  question={question!r}")
        with trace_span("retrieve", question=question, k=self.k):
            retrieved = self.retriever.pick(question, k=self.k)
        contexts = [r.chunk.text for r in retrieved]
        context_block = self._format_context_block(retrieved)

        # ── STAGE 2 — prompt ────────────────────────────────────────────
        print(f"[pipeline] step 2/3 — prompt  "
              f"({len(retrieved)} chunks, "
              f"{sum(len(c) for c in contexts)} chars)")
        prompt = USER_TEMPLATE.format(
            context_block=context_block, question=question)

        # ── STAGE 3 — generate (decorated child run) ───────────────────
        print(f"[pipeline] step 3/3 — generate  model={self.model}")
        answer, in_tok, out_tok, cost_usd = self._generate(prompt)

        return PipelineResult(
            question=question,
            answer=answer,
            contexts=contexts,
            retrieved=retrieved,
            cost_usd=cost_usd,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    @track(name="RAGPipeline.generate")
    def _generate(self, prompt: str) -> tuple[str, int, int, float]:
        """One LLM call. Wrapped as its own LangSmith child run so it shows
        up as a span in the trace tree."""
        try:
            client = self._get_client()
            resp = client.messages.create(
                model=self.model,
                max_tokens=ANSWER_MAX_TOKENS,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = "".join(blk.text for blk in resp.content
                              if getattr(blk, "type", None) == "text"
                              ).strip()
            usage = getattr(resp, "usage", None)
            in_tok = int(getattr(usage, "input_tokens", 0) or 0)
            out_tok = int(getattr(usage, "output_tokens", 0) or 0)
            cost = _offline_cost(self.model, in_tok, out_tok)
            # Attach to the offline run (no-op when LangSmith is active —
            # the platform already has the real numbers).
            set_current_run_metrics(input_tokens=in_tok,
                                     output_tokens=out_tok,
                                     cost_usd=cost)
            return answer, in_tok, out_tok, cost
        except Exception as e:
            print(f"[pipeline] WARN: generation failed ({e!s})")
            return f"[generation failed: {e!s}]", 0, 0, 0.0

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _format_context_block(retrieved: List[Retrieved]) -> str:
        if not retrieved:
            return "(no context retrieved)"
        lines: List[str] = []
        for i, r in enumerate(retrieved, start=1):
            lines.append(f"[Chunk {i}  src={r.chunk.source}  "
                         f"score={r.score:.3f}]")
            lines.append(r.chunk.text.strip())
            lines.append("")
        return "\n".join(lines).strip()


# ── A demo when the file is run directly ────────────────�