"""
Session 33 — ingest.py

Load the sample corpus into a Chroma collection.

This module is the bridge from Phase 3 (S19 introduced ChromaDB) to
Phase 5. The collection produced here is the SAME collection the
agentic loop will retrieve from. Students should be able to:

  python ingest.py                    # build the collection in ./chroma
  python ingest.py --inspect          # print collection stats
  USE_REAL_EMBEDDINGS=1 python ingest.py   # use sentence-transformers

What this file is NOT:
  - It is not a chunking benchmark (S21 already did that).
  - It is not a retrieval-quality eval (S25 already did that).
  - It is the simplest "load corpus, chunk, embed, store" pipeline that
    gives the agentic loop something to call.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from embeddings import embed_many, embedder_label


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


CORPUS_DIR = Path(__file__).parent / "corpus"

# Default lives next to this file. Override with S33_CHROMA_DIR for CI or
# for filesystems that don't honour SQLite locking (FUSE, some network
# mounts). The S32 SQLite store needed the same escape hatch.
_DEFAULT_CHROMA_DIR = Path(__file__).parent / "chroma"
CHROMA_DIR = Path(os.environ.get("S33_CHROMA_DIR") or _DEFAULT_CHROMA_DIR)
COLLECTION_NAME = "prepdeck_kb"

# A small chunk size keeps the demo readable. The whole corpus is < 4 KB
# so even small chunks fit comfortably in the vector store.
TARGET_CHUNK_CHARS = 480
CHUNK_OVERLAP_CHARS = 60


# ----------------------------------------------------------------------------
# Chunker — paragraph-aware, with a soft length target
# ----------------------------------------------------------------------------


@dataclass
class _RawChunk:
    chunk_id: str
    text: str
    source: str


def _paragraphs(text: str) -> List[str]:
    """Split a markdown doc on blank lines, keeping each paragraph intact."""

    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(path: Path) -> List[_RawChunk]:
    """Greedy paragraph-pack: glue paragraphs until we hit the target size.

    Why paragraphs and not fixed-size? Markdown documents have natural
    section boundaries. Cutting across them creates chunks like
    "ridiculous administrative fee. ## Contact". Keeping a paragraph
    intact preserves the local context the embedder needs.
    """

    text = path.read_text(encoding="utf-8")
    paras = _paragraphs(text)
    chunks: List[_RawChunk] = []
    buffer: List[str] = []
    buffer_len = 0
    offset = 0

    def _flush() -> None:
        nonlocal buffer, buffer_len, offset
        if not buffer:
            return
        joined = "\n\n".join(buffer)
        chunks.append(
            _RawChunk(
                chunk_id=f"{path.name}:{offset:04d}",
                text=joined,
                source=path.name,
            )
        )
        offset += 1
        # Carry overlap from the last paragraph so neighbouring chunks
        # share a little context — this helps retrieval on questions
        # that straddle a section break.
        if joined and CHUNK_OVERLAP_CHARS > 0:
            tail = joined[-CHUNK_OVERLAP_CHARS:]
            buffer = [tail]
            buffer_len = len(tail)
        else:
            buffer = []
            buffer_len = 0

    for para in paras:
        if buffer_len + len(para) + 2 > TARGET_CHUNK_CHARS and buffer:
            _flush()
        buffer.append(para)
        buffer_len += len(para) + 2

    _flush()
    return chunks


# ----------------------------------------------------------------------------
# Chroma loaders
# ----------------------------------------------------------------------------


def _get_chroma_collection(persist: bool = True):
    """Open (or create) the prepdeck_kb collection.

    Uses Chroma's PersistentClient so the collection survives across
    Python processes — the demos open and close it many times.
    """

    try:
        import chromadb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "chromadb not installed. Run 'pip install chromadb'."
        ) from exc

    if persist:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    else:
        client = chromadb.EphemeralClient()
    # get_or_create_collection plays nicely with re-ingest.
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return coll


def ingest(reset: bool = True, persist: bool = True) -> int:
    """Walk the corpus directory, chunk each doc, embed, upsert into Chroma.

    Returns the total number of chunks loaded.
    """

    coll = _get_chroma_collection(persist=persist)

    if reset:
        # Clear out anything left over from a previous run so re-ingest is
        # idempotent. delete() with no args clears the collection.
        existing = coll.get()
        if existing and existing.get("ids"):
            coll.delete(ids=existing["ids"])

    all_chunks: List[_RawChunk] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        all_chunks.extend(chunk_document(path))

    if not all_chunks:
        raise RuntimeError(f"No markdown files found in {CORPUS_DIR}.")

    # Embed in a single batch — the corpus is tiny.
    vectors = embed_many([c.text for c in all_chunks])

    coll.add(
        ids=[c.chunk_id for c in all_chunks],
        embeddings=vectors,
        documents=[c.text for c in all_chunks],
        metadatas=[{"source": c.source} for c in all_chunks],
    )

    return len(all_chunks)


def inspect() -> None:
    """Print a few stats about the live collection so demos can sanity-check."""

    coll = _get_chroma_collection(persist=True)
    snap = coll.get()
    ids = snap.get("ids", []) or []
    sources: List[str] = []
    for md in snap.get("metadatas", []) or []:
        if md and "source" in md:
            sources.append(md["source"])
    by_source: dict[str, int] = {}
    for s in sources:
        by_source[s] = by_source.get(s, 0) + 1

    print(f"Collection: {COLLECTION_NAME}")
    print(f"  embedder : {embedder_label()}")
    print(f"  chunks   : {len(ids)}")
    for src, n in sorted(by_source.items()):
        print(f"    - {src:35s} {n:>3} chunks")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="print collection stats and exit (does not re-ingest)",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="append to the existing collection instead of clearing it",
    )
    args = parser.parse_args(argv)

    if args.inspect:
        inspect()
        return 0

    n = ingest(reset=not args.no_reset)
    print(f"Ingested {n} chunks into '{COLLECTION_NAME}' via {embedder_label()}.")
    print(f"Persisted to: {CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
