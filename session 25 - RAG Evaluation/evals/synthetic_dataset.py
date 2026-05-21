"""
Session 25 — evals/synthetic_dataset.py

THE BOOTSTRAP — generate eval rows from your corpus when you don't yet
have a hand-curated golden dataset.

Ragas ships `TestsetGenerator` for exactly this. It reads your documents
and uses an LLM to produce (question, ground_truth, ground_truth_contexts)
triples. Useful when:

  - you're new to a corpus and need to bootstrap an eval in an hour
  - you want adversarial questions you wouldn't have thought of
  - you're stress-testing a retriever on a corpus you haven't read

NOT a replacement for human-curated rows on the questions that ACTUALLY
matter to your users. Synthetic data fills the breadth; hand-curated
data fills the depth.

OFFLINE FALLBACK
----------------
If `ragas` isn't installed, this module falls back to a simple
Anthropic-powered generator that produces synthetic rows directly from
the corpus text. Same output shape, smaller scale.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from loaders import Doc
from .golden_dataset import GoldenRow


load_dotenv()


DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# ── Public entry point ──────────────────────────────────────────────────────

def generate_synthetic_rows(docs: List[Doc],
                            *,
                            n_rows: int = 5,
                            model: str = DEFAULT_MODEL
                            ) -> List[GoldenRow]:
    """Produce `n_rows` synthetic (question, ground_truth, contexts) triples.

    Tries `ragas.testset.TestsetGenerator` first; falls back to a tiny
    Anthropic-only generator if Ragas isn't installed.
    """
    print(f"[synthetic] requesting {n_rows} synthetic eval rows...")
    try:
        return _generate_with_ragas(docs, n_rows=n_rows)
    except ImportError:
        print("[synthetic] WARN: ragas not installed — using the offline "
              "fallback generator.")
        return _generate_with_fallback(docs, n_rows=n_rows, model=model)
    except Exception as e:
        # Ragas API churn (TypeError on TestsetGenerator init, rate limits,
        # auth errors, etc.) — degrade gracefully to the offline Anthropic
        # generator instead of crashing the whole demo.
        print(f"[synthetic] WARN: ragas path failed ({type(e).__name__}: {e!s}) "
              "— using the offline fallback generator.")
        return _generate_with_fallback(docs, n_rows=n_rows, model=model)


# ── Real Ragas path ────────────────────────────────────────────────────────

def _generate_with_ragas(docs: List[Doc], *, n_rows: int) -> List[GoldenRow]:
    """Wrap `ragas.testset.TestsetGenerator`.

    Ragas 0.2.x changed `TestsetGenerator.from_langchain()` to require
    explicit `llm` and `embedding_model` arguments — same shape of fix
    we applied to harness_ragas.evaluate(): wrap a LangChain LLM +
    embedder in Ragas's adapter wrappers and pass them in.
    """
    from ragas.testset import TestsetGenerator
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # Convert our Doc dataclass into the shape Ragas wants. The exact
    # adapter changes across Ragas versions; we use the document-text path
    # because it works back to ragas 0.1.x.
    try:
        from langchain_core.documents import Document as LCDoc  # type: ignore
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import OpenAIEmbeddings

        lc_docs = [LCDoc(page_content=d.text,
                         metadata={"source": d.source, **d.metadata})
                   for d in docs]

        # Explicit judge + embedder wrapping for Ragas 0.2.x.
        # max_tokens=4096 — TestsetGenerator drafts full (Q, A, contexts)
        # triples per turn. A 512-cap truncates mid-paragraph and raises
        # `LLMDidNotFinishException`. 4096 is plenty for any single row.
        generator_llm = LangchainLLMWrapper(ChatAnthropic(
            model=DEFAULT_MODEL, max_tokens=4096, temperature=0.0))
        generator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small"))

        gen = TestsetGenerator.from_langchain(
            llm=generator_llm,
            embedding_model=generator_embeddings,
        )
        # Ragas 0.2.x renamed `test_size` to `testset_size` on this method.
        testset = gen.generate_with_langchain_docs(
            documents=lc_docs, testset_size=n_rows)
    except ImportError as e:
        # langchain_core / langchain_anthropic / langchain_openai not
        # installed. Ragas 0.2.x's TestsetGenerator() also can't be
        # instantiated without an LLM and embedding model, so the legacy
        # `gen.generate(...)` path no longer works either. Re-raise so the
        # outer `generate_synthetic_rows` handler can route to the offline
        # Anthropic-only fallback.
        raise ImportError(
            "Ragas requires langchain-core, langchain-anthropic, and "
            "langchain-openai for synthetic data generation. Install them "
            "or rely on the offline fallback."
        ) from e

    df = testset.to_pandas()
    rows: List[GoldenRow] = []
    for _, r in df.iterrows():
        rows.append(GoldenRow(
            question=str(r.get("question", "")),
            ground_truth=str(r.get("ground_truth",
                                   r.get("reference", ""))),
            ground_truth_contexts=list(r.get("reference_contexts",
                                             r.get("contexts", []))) or [],
            metadata={"synthetic": True, "source": "ragas"},
        ))
    print(f"[synthetic] ragas produced {len(rows)} rows")
    return rows


# ── Offline fallback ───────────────────────────────────────────────────────

_GEN_PROMPT = (
    "You are a test-set generator. Given a document, write exactly ONE "
    "natural question a user might ask, ONE precise answer drawn from the "
    "document, and ONE single sentence quoted from the document that "
    "supports the answer.\n\n"
    "Document:\n{document}\n\n"
    "Return a JSON object with the keys 'question', 'answer', "
    "'evidence_sentence'. JSON only, no markdown fence, no comment.\n"
    "Example:\n"
    "{{\"question\": \"...\", \"answer\": \"...\", \"evidence_sentence\": \"...\"}}"
)


_anthropic_client: Any = None


def _client() -> Any:
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError("anthropic package not installed") from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def _strip_json(text: str) -> Dict[str, Any]:
    """Pull a JSON object out of the LLM reply even if it wrapped it
    in a markdown fence or added prose around it."""
    # Strip ```json ... ``` if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if not brace:
            return {}
        candidate = brace.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


def _generate_with_fallback(docs: List[Doc],
                            *,
                            n_rows: int,
                            model: str) -> List[GoldenRow]:
    """Round-robin docs and ask Claude for a (Q, A, evidence) triple."""
    if not docs:
        return []
    rows: List[GoldenRow] = []
    cursor = 0
    while len(rows) < n_rows and cursor < n_rows * 3:
        doc = docs[cursor % len(docs)]
        prompt = _GEN_PROMPT.format(document=doc.text[:3000])
        try:
            resp = _client().messages.create(
                model=model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(blk.text for blk in resp.content
                           if getattr(blk, "type", None) == "text")
            obj = _strip_json(text)
        except Exception as e:
            print(f"[synthetic-fallback] WARN: generation failed ({e!s})")
            cursor += 1
            continue

        if not obj or "question" not in obj:
            cursor += 1
            continue

        rows.append(GoldenRow(
            question=str(obj.get("question", "")).strip(),
            ground_truth=str(obj.get("answer", "")).strip(),
            ground_truth_contexts=[
                str(obj.get("evidence_sentence", "")).strip()
            ] if obj.get("evidence_sentence") else [],
            metadata={"synthetic": True,
                      "source": "offline-fallback",
                      "from": doc.source},
        ))
        print(f"[synthetic-fallback] row {len(rows)}/{n_rows}  "
              f"Q={obj.get('question', '')[:60]!r}")
        cursor += 1

    return rows


if __name__ == "__main__":
    import sys
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, HERE)
    from loaders import load_corpus

    files = [
        os.path.join(HERE, "data", "intro_to_rag.pdf"),
        os.path.join(HERE, "data", "product_manual.pdf"),
    ]
    docs = load_corpus(files)
    rows = generate_synthetic_rows(docs, n_rows=3)
    print(f"\n[synthetic] produced {len(rows)} synthetic rows:")
    for r in rows:
        print(f"  Q: {r.question}")
        print(f"  A: {r.ground_truth[:80]}...")
        print()
