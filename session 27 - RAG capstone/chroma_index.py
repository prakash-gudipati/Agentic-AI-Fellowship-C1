"""
Session 27 -- chroma_index.py

CHROMADB-BACKED CHUNKINDEX -- promoted from "S19 cameo" to production
default in S27. The capstone is the first time the curriculum's
reference code wires ChromaDB end-to-end.

WHY THIS FILE
-------------
S20-S26 used the in-memory `ChunkIndex` from `retrievers/base.py` --
a list of Chunks plus a numpy matrix of L2-normalised vectors. That
works for the fellowship demo corpus (~30 chunks). It does NOT work
for the capstone, where students bring their own corpora of hundreds
to thousands of chunks across multiple sessions.

ChromaDB gives us:
  - **persistent storage between runs** -- the killer feature at
    capstone scale. Re-runs against the same corpus do zero embed
    calls.
  - similarity search optimised for large collections.
  - metadata filtering (we keep S22's `filter_by` semantics).
  - a tested, MIT-licensed on-disk store the student can ship.

DROP-IN COMPATIBILITY
---------------------
`ChromaChunkIndex` exposes the SAME public surface as the in-memory
`ChunkIndex` from S22:

  .chunks  -> List[Chunk]
  .vectors -> np.ndarray  (L2-normalised)
  .add(chunks)
  __len__()

That means every retriever in `retrievers/` -- SimilarityRetriever
from S22, MMR / hybrid / reranker from S23 -- works against the
ChromaDB-backed index without any modification. The retriever
strategy pattern earns its keep one more time.

WHAT CHROMADB ADDS ON TOP
-------------------------
Persistence. On first `add()`, vectors land in `.chroma/<collection>/`.
On the next process boot, the constructor reads the collection back
into the in-memory matrix WITHOUT issuing a single embed call. That
is the capstone-scale win.

OFFLINE FALLBACK
----------------
If `chromadb` isn't installed, `is_chromadb_available()` returns
False; the capstone's `demo.py` reads that flag and falls back to the
in-memory `ChunkIndex` so the walkthrough still runs.

PRODUCTION PATTERNS REINFORCED (S27):
  - protocol-based abstraction (any backend matching the ChunkIndex
    public surface plugs in)              (S22 reinforced)
  - embed-once, query-many                (S25 reinforced)
  - persistence as a first-class concern  (S27 NEW)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from chunker import Chunk
from embeddings import embed_texts, l2_normalise


DEFAULT_COLLECTION = "fellowship-capstone"
DEFAULT_PERSIST_DIR = ".chroma"


def is_chromadb_available() -> bool:
    """True iff the `chromadb` package can be imported. Useful for the
    capstone walkthrough to print the active backend at startup."""
    try:
        import chromadb                              # noqa: F401
        return True
    except ImportError:
        return False


class ChromaChunkIndex:
    """ChromaDB-backed implementation of the ChunkIndex public surface.

    DROP-IN for `retrievers.base.ChunkIndex` -- exposes `.chunks` and
    `.vectors` so the existing retrievers work unchanged.

    PERSISTENCE -- writes to `<persist_dir>/<collection>/`. On next
    boot, vectors + chunks load WITHOUT re-embedding.
    """

    def __init__(self,
                 *,
                 collection: str = DEFAULT_COLLECTION,
                 persist_dir: str = DEFAULT_PERSIST_DIR) -> None:
        try:
            import chromadb                          # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "chromadb is not installed. The capstone falls back "
                "to the in-memory ChunkIndex in this case. To use the "
                "production default, run:  pip install chromadb"
            ) from e

        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection)
        self.collection_name = collection
        self.persist_dir = os.path.abspath(persist_dir)

        # Drop-in ChunkIndex surface.
        self.chunks: List[Chunk] = []
        self.vectors: np.ndarray = np.zeros((0, 1536), dtype=np.float32)

        # Re-hydrate from any persisted rows.
        self._reload_from_collection()

    # ── Drop-in ChunkIndex API ────────────────────────────────────────

    def add(self, chunks: List[Chunk]) -> None:
        """Embed + L2-normalise + append + persist to Chroma.

        Skips chunks whose deterministic id is already in the
        collection -- so re-runs against the same corpus do zero
        embed calls AND zero Chroma writes."""
        if not chunks:
            return

        new_chunks: List[Chunk] = []
        new_ids: List[str] = []
        new_texts: List[str] = []

        existing = self._existing_ids()
        for c in chunks:
            cid = self._chunk_id(c)
            if cid in existing:
                continue
            new_chunks.append(c)
            new_ids.append(cid)
            new_texts.append(c.text)

        if not new_chunks:
            print(f"[chroma] collection {self.collection_name!r} "
                  f"already has all {len(chunks)} chunks; nothing to "
                  f"embed.")
            return

        # Embed only the new ones, L2-normalise, append to in-memory
        # mirror + persist to Chroma.
        new_vecs = embed_texts(new_texts)
        new_vecs = l2_normalise(new_vecs)

        # Append to in-memory mirror.
        self.chunks.extend(new_chunks)
        self.vectors = (new_vecs if self.vectors.shape[0] == 0
                        else np.vstack([self.vectors, new_vecs]))

        # Persist to Chroma.
        metadatas = []
        for c in new_chunks:
            md = dict(c.metadata or {})
            md["source"] = c.source
            md["chunk_id"] = c.chunk_id
            metadatas.append({k: _coerce_md(v) for k, v in md.items()})

        self._collection.add(
            ids=new_ids,
            embeddings=new_vecs.tolist(),
            documents=new_texts,
            metadatas=metadatas,
        )
        print(f"[chroma] embedded + persisted {len(new_chunks)} new "
              f"chunks. Collection now holds {len(self.chunks)} "
              f"(matrix shape={self.vectors.shape}).")

    def __len__(self) -> int:
        return len(self.chunks)

    # ── ChromaDB-specific helpers ─────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "backend":         "chromadb",
            "collection":      self.collection_name,
            "persist_dir":     self.persist_dir,
            "embedded_chunks": len(self.chunks),
            "vector_dim":      self.vectors.shape[1]
                                if self.vectors.size else 0,
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _reload_from_collection(self) -> None:
        """Pull existing rows into the in-memory matrix. Zero embed
        calls. Runs once at construction."""
        try:
            got = self._collection.get(
                include=["documents", "metadatas", "embeddings"])
        except Exception as e:
            print(f"[chroma] WARN: reload failed ({e!s}); starting empty.")
            return
        ids = got.get("ids") or []
        if not ids:
            return
        docs = got.get("documents") or []
        mds = got.get("metadatas") or []
        embs = got.get("embeddings") or []

        rebuilt_chunks: List[Chunk] = []
        rebuilt_vectors: List[List[float]] = []
        for i, cid in enumerate(ids):
            md = mds[i] or {}
            chunk = Chunk(
                text=docs[i] if i < len(docs) else "",
                source=str(md.get("source", "?")),
                chunk_id=int(md.get("chunk_id", 0)),
                metadata={k: v for k, v in md.items()
                          if k not in ("source", "chunk_id")},
            )
            rebuilt_chunks.append(chunk)
            if i < len(embs) and embs[i] is not None:
                rebuilt_vectors.append(list(embs[i]))

        self.chunks = rebuilt_chunks
        if rebuilt_vectors:
            self.vectors = np.asarray(rebuilt_vectors, dtype=np.float32)
            # Defensive: ensure rows are unit-norm.
            self.vectors = l2_normalise(self.vectors)
        print(f"[chroma] re-hydrated {len(self.chunks)} chunks from "
              f"{self.collection_name!r} (zero embed calls).")

    def _existing_ids(self) -> set:
        try:
            got = self._collection.get()
            return set(got.get("ids") or [])
        except Exception:
            return set()

    def _chunk_id(self, c: Chunk) -> str:
        return f"{c.source}::{c.chunk_id}"


# ── Module-level helpers ──────────────────────────────────────────────

def _coerce_md(value: Any) -> Any:
    """ChromaDB metadata values must be primitive types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    print(f"[chroma] is_chromadb_available() = {is_chromadb_available()}")
    if not is_chromadb_available():
        print("[chroma] install with: pip install chromadb")
        raise SystemExit(0)

    # Tiny smoke test against an in-memory list of chunks.
    chunks = [
        Chunk(text="RAG augments LLMs with retrieved context.",
              source="demo.txt", chunk_id=0, metadata={}),
        Chunk(text="Embeddings turn text into numbers.",
              source="demo.txt", chunk_id=1, metadata={}),
        Chunk(text="Vector databases store and query embeddings at scale.",
              source="demo.txt", chunk_id=2, metadata={}),
    ]
    idx = ChromaChunkIndex(collection="chroma-smoke",
                           persist_dir=".chroma-smoke")
    idx.add(chunks)
    print(f"\nstats: {idx.stats()}")
    print(f"len:   {len(idx)}")
    print(f"vec0:  shape={idx.vectors[0].shape}  "
          f"first_5={idx.vectors[0][:5].tolist()}")
