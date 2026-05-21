"""
Session 23 — chunkers.py

Five chunking strategies behind a single function-level interface, so the
25-combination matrix runner can iterate over them.

The five strategies (kept deliberately simple — chunking is not today's
lesson, the matrix is):

  1. fixed       — slice every N characters
  2. sentence    — split on sentence boundaries, pack until size budget hit
  3. recursive   — paragraph → sentence → character; respects natural boundaries
                   (the S22 default; the LangChain default)
  4. structural  — split on document-level headers (Markdown-ish detection)
  5. semantic    — embed sentences, cut where adjacent-sentence similarity drops

Each chunker exposes the same  chunk(docs) → List[Chunk]  contract.

The PUBLIC chunker for non-matrix use stays  chunker.chunk_corpus(docs),
which uses the recursive strategy and is unchanged from S22.

Production patterns reinforced:
  - chunker strategy pattern (S21)
  - one Chunk shape for every retriever (S22)
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List

import numpy as np

from chunker import Chunk
from loaders import Doc


# ── Defaults — match the S22 chunker so behaviour is consistent ─────────────
DEFAULT_SIZE = 500
DEFAULT_OVERLAP = 80

# Semantic-chunker threshold — split when adjacent-sentence cosine drops below
# this. Hand-tuned for the small Phase 4 corpus.
SEMANTIC_DROP = 0.65


# ── Helpers ─────────────────────────────────────────────────────────────────

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"\'])")
_HEADER_LINE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 \-]{4,}$|\d+(\.\d+)*\s+[A-Z].+)$",
                          flags=re.MULTILINE)


def _split_sentences(text: str) -> List[str]:
    """Cheap sentence splitter. Production: use spaCy or NLTK."""
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── Strategy 1 — fixed ──────────────────────────────────────────────────────

def chunk_fixed(docs: List[Doc],
                size: int = DEFAULT_SIZE,
                overlap: int = DEFAULT_OVERLAP) -> List[Chunk]:
    """Slice every N characters with overlap. The bluntest strategy."""
    out: List[Chunk] = []
    cid = 0
    step = max(size - overlap, 1)
    for d in docs:
        n = 0
        for i in range(0, len(d.text), step):
            piece = d.text[i:i + size].strip()
            if piece:
                out.append(Chunk(chunk_id=cid, source=d.source,
                                 text=piece, metadata=dict(d.metadata)))
                cid += 1
                n += 1
        print(f"[chunkers:fixed]      {d.source:<24} → {n:>3} chunks")
    return out


# ── Strategy 2 — sentence ───────────────────────────────────────────────────

def chunk_sentence(docs: List[Doc],
                   size: int = DEFAULT_SIZE,
                   overlap: int = DEFAULT_OVERLAP) -> List[Chunk]:
    """Pack sentences into groups until the size budget is hit."""
    out: List[Chunk] = []
    cid = 0
    for d in docs:
        sentences = _split_sentences(d.text)
        if not sentences:
            continue
        buf: List[str] = []
        buf_len = 0
        n = 0
        for s in sentences:
            if buf_len + len(s) > size and buf:
                out.append(Chunk(chunk_id=cid, source=d.source,
                                 text=" ".join(buf).strip(),
                                 metadata=dict(d.metadata)))
                cid += 1
                n += 1
                # Overlap — keep the trailing sentences whose total length
                # is roughly `overlap`.
                tail: List[str] = []
                tail_len = 0
                for prev in reversed(buf):
                    if tail_len + len(prev) > overlap:
                        break
                    tail.append(prev)
                    tail_len += len(prev)
                buf = list(reversed(tail))
                buf_len = sum(len(t) + 1 for t in buf)
            buf.append(s)
            buf_len += len(s) + 1
        if buf:
            out.append(Chunk(chunk_id=cid, source=d.source,
                             text=" ".join(buf).strip(),
                             metadata=dict(d.metadata)))
            cid += 1
            n += 1
        print(f"[chunkers:sentence]   {d.source:<24} → {n:>3} chunks")
    return out


# ── Strategy 3 — recursive ──────────────────────────────────────────────────

def chunk_recursive(docs: List[Doc],
                    size: int = DEFAULT_SIZE,
                    overlap: int = DEFAULT_OVERLAP) -> List[Chunk]:
    """Paragraph → sentence → character. Respects natural boundaries.

    Identical-in-spirit to the S22 default and the LangChain default.
    """
    out: List[Chunk] = []
    cid = 0
    for d in docs:
        pieces: List[str] = []
        for para in d.text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para) <= size:
                pieces.append(para)
                continue
            # Paragraph too long — try sentences.
            sents = _split_sentences(para)
            buf: List[str] = []
            buf_len = 0
            for s in sents:
                if buf_len + len(s) > size and buf:
                    pieces.append(" ".join(buf))
                    buf = []
                    buf_len = 0
                if len(s) > size:
                    # Sentence too long — character window with overlap.
                    for i in range(0, len(s), size - overlap):
                        pieces.append(s[i:i + size])
                else:
                    buf.append(s)
                    buf_len += len(s) + 1
            if buf:
                pieces.append(" ".join(buf))
        n = 0
        for p in pieces:
            p = p.strip()
            if not p:
                continue
            out.append(Chunk(chunk_id=cid, source=d.source,
                             text=p, metadata=dict(d.metadata)))
            cid += 1
            n += 1
        print(f"[chunkers:recursive]  {d.source:<24} → {n:>3} chunks")
    return out


# ── Strategy 4 — structural (header-aware) ──────────────────────────────────

def chunk_structural(docs: List[Doc],
                     size: int = DEFAULT_SIZE,
                     overlap: int = DEFAULT_OVERLAP) -> List[Chunk]:
    """Split on detected headers, then size-cap each section."""
    out: List[Chunk] = []
    cid = 0
    for d in docs:
        # Find header offsets. Split text into sections between them.
        positions = [m.start() for m in _HEADER_LINE.finditer(d.text)]
        if not positions:
            # No headers detected — fall through to recursive on the whole doc.
            sub_chunks = chunk_recursive([d], size=size, overlap=overlap)
            for sc in sub_chunks:
                sc.chunk_id = cid
                cid += 1
                out.append(sc)
            print(f"[chunkers:structural] {d.source:<24} → "
                  f"{len(sub_chunks):>3} chunks (no headers)")
            continue

        positions.append(len(d.text))
        sections: List[str] = []
        for a, b in zip(positions, positions[1:]):
            sec = d.text[a:b].strip()
            if sec:
                sections.append(sec)

        n = 0
        for sec in sections:
            if len(sec) <= size:
                out.append(Chunk(chunk_id=cid, source=d.source,
                                 text=sec, metadata=dict(d.metadata)))
                cid += 1
                n += 1
                continue
            # Long section — recursive-chunk just this slice.
            sub = chunk_recursive(
                [Doc(source=d.source, text=sec, metadata=d.metadata)],
                size=size, overlap=overlap,
            )
            for sc in sub:
                sc.chunk_id = cid
                cid += 1
                n += 1
                out.append(sc)
        print(f"[chunkers:structural] {d.source:<24} → {n:>3} chunks")
    return out


# ── Strategy 5 — semantic (boundary by similarity drop) ─────────────────────

def chunk_semantic(docs: List[Doc],
                   size: int = DEFAULT_SIZE,
                   overlap: int = DEFAULT_OVERLAP,
                   drop_threshold: float = SEMANTIC_DROP) -> List[Chunk]:
    """Embed sentences. Cut where adjacent cosine drops below threshold.

    Lazy-imports embeddings so the matrix runner can be loaded even when
    no API key is set (only this chunker actually needs the embedding API).
    """
    from embeddings import embed_texts, l2_normalise

    out: List[Chunk] = []
    cid = 0
    for d in docs:
        sentences = _split_sentences(d.text)
        if len(sentences) <= 1:
            if d.text.strip():
                out.append(Chunk(chunk_id=cid, source=d.source,
                                 text=d.text.strip(),
                                 metadata=dict(d.metadata)))
                cid += 1
            continue

        vecs = l2_normalise(embed_texts(sentences))
        # Cut indices — boundaries where adjacent cosine < threshold.
        cuts: List[int] = []
        for i in range(len(sentences) - 1):
            sim = float(vecs[i] @ vecs[i + 1])
            if sim < drop_threshold:
                cuts.append(i + 1)
        cuts = [0] + cuts + [len(sentences)]

        groups: List[List[str]] = [
            sentences[a:b] for a, b in zip(cuts, cuts[1:])
        ]
        n = 0
        for grp in groups:
            buf = " ".join(grp).strip()
            if not buf:
                continue
            # Enforce upper bound — fall through to fixed-size if a group is huge.
            if len(buf) <= size * 1.5:
                out.append(Chunk(chunk_id=cid, source=d.source,
                                 text=buf, metadata=dict(d.metadata)))
                cid += 1
                n += 1
            else:
                for i in range(0, len(buf), size - overlap):
                    out.append(Chunk(chunk_id=cid, source=d.source,
                                     text=buf[i:i + size],
                                     metadata=dict(d.metadata)))
                    cid += 1
                    n += 1
        print(f"[chunkers:semantic]   {d.source:<24} → {n:>3} chunks "
              f"(drop={drop_threshold})")
    return out


# ── Registry ────────────────────────────────────────────────────────────────

# Order matters — this is the order they appear as rows of the matrix.
CHUNKERS: Dict[str, Callable[[List[Doc]], List[Chunk]]] = {
    "fixed":      chunk_fixed,
    "sentence":   chunk_sentence,
    "recursive":  chunk_recursive,
    "structural": chunk_structural,
    "semantic":   chunk_semantic,
}


if __name__ == "__main__":
    import os
    from loaders import load_corpus
    here = os.path.dirname(__file__)
    docs = load_corpus([
        os.path.join(here, "data", "intro_to_rag.pdf"),
        os.path.join(here, "data", "product_manual.pdf"),
        os.path.join(here, "data", "sample_article.html"),
    ])
    print()
    for name, fn in CHUNKERS.items():
        if name == "semantic":
            # Skip semantic in the smoke test unless an API key is set.
            if not os.environ.get("OPENAI_API_KEY"):
                print(f"[smoke] skipping semantic (no OPENAI_API_KEY)")
                continue
        chunks = fn(docs)
        print(f"  {name:<11} →  total {len(chunks)} chunks")
