"""
Session 25 — pipeline.py

The end-to-end RAG callable that today's evals score.

Why does this exist as its own module?
  - Eval frameworks like Ragas need a single function that takes
    a question and returns BOTH the answer AND the retrieved contexts.
    A callable pipeline IS the unit under test.
  - The harness wraps THIS, not the individual retriever or LLM. That's
    the right granularity: a real user judges the end-to-end output, not
    the cosine score.

Architecture:

    question  →  retriever.pick(question)  →  contexts
                                          ↓
                          messages = [system + contexts + question]
                                          ↓
                                       LLM call
                                          ↓
                                    (answer, contexts)

The retriever is INJECTED — you can plug similarity, hybrid (S23),
contextual (S24), or any other retriever that conforms to the S22
strategy pattern. The rest of the pipeline doesn't change. That's the
strategy pattern paying off again.

Production patterns reinforced (S25):
  - dependency injection of the retriever      (S22)
  - structured return type (NamedTuple)        (S22)
  - try/except on every external API call      (S3)
  - env var for API key, never hard-coded      (S8)
  - explicit prompt with citation discipline   (S14)
  - pipeline logging with step numbers         (S20)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from retrievers.base import Retrieved, Retriever


load_dotenv()

# ── Defaults ────────────────────────────────────────────────────────────────
ANSWER_MODEL = "claude-haiku-4-5-20251001"   # cheap + fast, good enough for eval
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
    "Context:\n"
    "{context_block}\n\n"
    "Question: {question}\n\n"
    "Answer using only the context above. Be specific and concise."
)


# ── Public return type ──────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """One end-to-end run of the pipeline."""
    question: str
    answer: str
    contexts: List[str]           # the chunk texts the LLM actually saw
    retrieved: List[Retrieved]    # the full Retrieved objects (incl. scores)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "retrieved_sources": [r.chunk.source for r in self.retrieved],
            "retrieved_scores": [round(r.score, 4) for r in self.retrieved],
        }


# ── The pipeline ────────────────────────────────────────────────────────────

class RAGPipeline:
    """Question → retrieved chunks → LLM answer.

    Construct once with a chosen retriever; call .answer(question) any
    number of times. The eval harness runs this in a loop over the
    golden dataset.
    """

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
        self._client = client  # lazy-init in _get_client()

    # ── External resources (lazy) ──────────────────────────────────────────

    def _get_client(self) -> Any:
        """Lazy-init the Anthropic client so importing doesn't need a key."""
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError("anthropic package not installed. "
                               "pip install anthropic") from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env "
                "and add your key."
            )
        self._client = Anthropic(api_key=api_key)
        return self._client

    # ── Main entry point ───────────────────────────────────────────────────

    def answer(self, question: str) -> PipelineResult:
        """Retrieve, prompt, generate. Returns a structured PipelineResult."""
        print(f"\n[pipeline] step 1/3 — retrieve  question={question!r}")
        retrieved = self.retriever.pick(question, k=self.k)

        contexts = [r.chunk.text for r in retrieved]
        context_block = self._format_context_block(retrieved)
        print(f"[pipeline] step 2/3 — prompt  "
              f"({len(retrieved)} chunks, "
              f"{sum(len(c) for c in contexts)} chars)")

        prompt = USER_TEMPLATE.format(
            context_block=context_block, question=question)

        print(f"[pipeline] step 3/3 — generate  model={self.model}")
        try:
            client = self._get_client()
            resp = client.messages.create(
                model=self.model,
                max_tokens=ANSWER_MAX_TOKENS,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = "".join(blk.text for blk in resp.content
                             if getattr(blk, "type", None) == "text").strip()
        except Exception as e:
            answer = f"[generation failed: {e!s}]"
            print(f"[pipeline] WARN: generation failed ({e!s})")

        return PipelineResult(
            question=question,
            answer=answer,
            contexts=contexts,
            retrieved=retrieved,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _format_context_block(retrieved: List[Retrieved]) -> str:
        """Inline the chunks with explicit numbering so the model can cite."""
        if not retrieved:
            return "(no context retrieved)"
        lines: List[str] = []
        for i, r in enumerate(retrieved, start=1):
            lines.append(f"[Chunk {i}  src={r.chunk.source}  "
                         f"score={r.score:.3f}]")
            lines.append(r.chunk.text.strip())
            lines.append("")
        return "\n".join(lines).strip()


# ── A demo when the file is run directly ───────────────────────────────────

if __name__ == "__main__":
    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)

    from loaders import load_corpus
    from chunker import chunk_corpus
    from embedding_cache import CachedChunkIndex, EmbeddingCache
    from retrievers.similarity import SimilarityRetriever

    files = [
        os.path.join(HERE, "data", "intro_to_rag.pdf"),
        os.path.join(HERE, "data", "product_manual.pdf"),
        os.path.join(HERE, "data", "sample_article.html"),
    ]
    docs = load_corpus(files)
    chunks = chunk_corpus(docs)

    cache = EmbeddingCache()
    index = CachedChunkIndex(cache=cache)
    index.add(chunks)

    retriever = SimilarityRetriever(index)
    pipe = RAGPipeline(retriever=retriever, k=3)

    result = pipe.answer("What is retrieval-augmented generation?")
    print("\n========= ANSWER =========")
    print(result.answer)
    print("==========================")
    print("retrieved sources:", [r.chunk.source for r in result.retrieved])

