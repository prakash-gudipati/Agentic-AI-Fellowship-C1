import os
from typing import List

from dotenv import load_dotenv
from anthropic import Anthropic

from retriever import Retrieved

# ── Configuration ───────────────────────────────────────────────────────────
GENERATION_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 600

GROUNDED_SYSTEM_PROMPT = """You are a careful research assistant.

Answer the user's question using ONLY the information in the CONTEXT block below.

Rules:
1. If the answer is in the context, answer in 2-4 sentences and cite the source filename(s) in square brackets at the end of each claim, like this: [intro_to_rag.pdf].
2. If the answer is NOT in the context, reply exactly: "I don't know based on the provided documents."
3. Do not use any prior knowledge. Do not speculate.
4. Do not invent source names — only cite filenames that actually appear in the CONTEXT block."""

load_dotenv()
_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def build_context_block(retrieved: List[Retrieved]) -> str:
    """Format retrieved chunks into a single CONTEXT block.

    Each chunk is labelled with its source filename so the model can cite it.
    """
    lines = []
    for i, r in enumerate(retrieved, start=1):
        lines.append(f"[Chunk {i} | source: {r.chunk.source} | score: {r.score:.3f}]")
        lines.append(r.chunk.text)
        lines.append("")
    return "\n".join(lines).strip()


def generate(query: str, retrieved: List[Retrieved]) -> str:
    """Compose the grounded prompt, call Claude, return the answer text."""
    if not retrieved:
        return "I don't know based on the provided documents."

    context_block = build_context_block(retrieved)
    user_message = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer the question using only the CONTEXT above."
    )

    client = _get_client()
    try:
        resp = client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=MAX_TOKENS,
            system=GROUNDED_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as e:
        raise RuntimeError(f"Anthropic generation call failed: {e}") from e

    answer = "".join(b.text for b in resp.content if b.type == "text").strip()
    print(f"[generator]  model={GENERATION_MODEL}  "
          f"input_tokens={resp.usage.input_tokens}  "
          f"output_tokens={resp.usage.output_tokens}")
    return answer


if __name__ == "__main__":
    # Quick smoke test — fakes a retrieved chunk and calls Claude.
    from chunker import Chunk
    fake = [
        Retrieved(
            chunk=Chunk(chunk_id=0, source="rag.txt",
                        text="RAG stands for Retrieval-Augmented Generation. "
                             "It combines a retrieval step with an LLM."),
            score=0.92,
        )
    ]
    print(generate("What does RAG stand for?", fake))
