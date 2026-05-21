"""
Session 25 — embedding_cache.py

THE PATTERN — cache embeddings by content hash.

WHY THIS EXISTS
---------------
In Phase 4 dev work you re-run the same corpus dozens of times. Each
re-run, without a cache, re-embeds every chunk from scratch. With a
100k-chunk corpus and OpenAI's text-embedding-3-small at ~$0.02 per
million tokens, every full re-run costs real money AND real seconds.

The cache stops that. It keys every vector on:

    (model_name, sha256(chunk_text))

Same model, same text in → same vector out, every time. If the text
hasn't changed, the cache hits and we skip the API call entirely. Only
NEW or CHANGED chunks pay.

Why the hash and not the chunk_id? Because chunk_id is unstable across
runs (re-chunk the corpus with a different size → all the ids shift),
but sha256 of the chunk TEXT is stable across runs, machines, and
collaborators. Two engineers on the same corpus share their caches.

WHY DISK
--------
A process-lifetime dict cache would die at every restart, defeating
the purpose. We store as one JSON file per hash inside .embedding_cache/
so the cache survives:

  - process restarts
  - branch switches
  - laptop reboots
  - sharing the cache directory across teammates

THE STATS COUNTER
-----------------
.hits and .misses are exposed so the demo can print exactly how many
calls were avoided. This single number is the most convincing
argument for the pattern when defending the code in a code review.

Production patterns introduced (S25):
  - cache by content hash                 (NEW, named today)
  - disk-backed cache (survives restarts) (NEW, named today)
  - read-through cache wrapping a raw embedder (NEW, named today)

Production patterns reinforced:
  - separate happy path from error path   (try/except on every call)
  - explicit log lines per stage          ([cache] prefix)
  - no hidden state on import             (lazy cache directory creation)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from embeddings import EMBEDDING_MODEL, embed_texts, l2_normalise


# ── Public API ──────────────────────────────────────────────────────────────
DEFAULT_CACHE_DIR = os.path.join(_HERE, ".embedding_cache")


def _sha256(text: str) -> str:
    """SHA-256 hex digest of the chunk text. Stable across machines."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Disk-backed cache of (model, sha256(text)) → np.ndarray.

    Each entry is stored as a tiny JSON file under cache_dir/<model>/<hash>.json
    containing the float list. The model name partitions the cache, so swapping
    embedding models does NOT silently mix old vectors with new ones.
    """

    def __init__(self,
                 cache_dir: str = DEFAULT_CACHE_DIR,
                 model: str = EMBEDDING_MODEL) -> None:
        self.cache_dir = cache_dir
        self.model = model
        self.model_dir = os.path.join(cache_dir, _safe_filename(model))
        os.makedirs(self.model_dir, exist_ok=True)

        # In-process metrics. Useful for the demo print-out at the end.
        self.hits = 0
        self.misses = 0
        # A small in-memory layer on top of disk, to avoid re-reading the same
        # vector twice within a single process.
        self._memory: Dict[str, np.ndarray] = {}

    # ── Core operations ────────────────────────────────────────────────────

    def get(self, text: str) -> Optional[np.ndarray]:
        """Return the cached vector for `text`, or None if not cached."""
        key = _sha256(text)
        if key in self._memory:
            self.hits += 1
            return self._memory[key]

        path = os.path.join(self.model_dir, f"{key}.json")
        if not os.path.exists(path):
            self.misses += 1
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            vector = np.asarray(payload["vector"], dtype=np.float32)
        except Exception as e:
            # Corrupt cache entry — treat as miss and overwrite later.
            print(f"[cache] WARN: failed to read {path} ({e!s}); treating as miss")
            self.misses += 1
            return None

        self._memory[key] = vector
        self.hits += 1
        return vector

    def put(self, text: str, vector: np.ndarray) -> None:
        """Persist `vector` under sha256(text). Overwrites if present."""
        key = _sha256(text)
        self._memory[key] = vector

        path = os.path.join(self.model_dir, f"{key}.json")
        payload = {"model": self.model, "vector": vector.tolist()}
        # Write to a temp file then rename, so a half-written file never
        # survives a crash. This is the canonical atomic-write pattern.
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp_path, path)
        except Exception as e:
            print(f"[cache] WARN: failed to write {path} ({e!s})")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ── Convenience ────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """Hit/miss counters since this process started."""
        total = self.hits + self.misses
        rate = (self.hits / total) if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": round(rate, 3),
        }

    def __len__(self) -> int:
        """Number of entries currently on disk for this model."""
        try:
            return sum(1 for n in os.listdir(self.model_dir)
                       if n.endswith(".json"))
        except FileNotFoundError:
            return 0


def _safe_filename(name: str) -> str:
    """Make a model name safe to use as a directory name."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


# ── The read-through wrapper everyone else imports ──────────────────────────

def embed_texts_cached(texts: List[str],
                       cache: EmbeddingCache) -> np.ndarray:
    """Drop-in replacement for embed_texts() that consults the cache first.

    For each text:
      1. Look up in cache. If hit, reuse the stored vector.
      2. If miss, queue for a batch embedding call.
    After the loop, embed all misses in one batched API call and store them.

    The returned matrix is in the SAME ORDER as the input — callers don't
    need to know the cache was involved.
    """
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)

    n = len(texts)
    result: List[Optional[np.ndarray]] = [None] * n
    miss_indices: List[int] = []
    miss_texts: List[str] = []

    for i, t in enumerate(texts):
        v = cache.get(t)
        if v is not None:
            result[i] = v
        else:
            miss_indices.append(i)
            miss_texts.append(t)

    if miss_texts:
        print(f"[cache] {n - len(miss_texts)} hits, {len(miss_texts)} misses  "
              f"→ embedding misses now")
        miss_matrix = embed_texts(miss_texts)
        for idx, vec in zip(miss_indices, miss_matrix):
            result[idx] = vec
            cache.put(texts[idx], vec)
    else:
        print(f"[cache] {n} hits, 0 misses  →  no API call needed")

    # Type-narrow: by construction every slot is now filled.
    return np.vstack([r for r in result if r is not None]).astype(np.float32)


# ── A ChunkIndex that uses the cache transparently ──────────────────────────
# We re-import here to avoid forcing every caller to know about the cache.

from chunker import Chunk  # noqa: E402
from retrievers.base import ChunkIndex  # noqa: E402


class CachedChunkIndex(ChunkIndex):
    """ChunkIndex variant that embeds via the cache.

    Same public interface as ChunkIndex. Drop in wherever ChunkIndex would
    have been used — no other code has to change.
    """

    def __init__(self, cache: Optional[EmbeddingCache] = None) -> None:
        super().__init__()
        self.cache = cache or EmbeddingCache()

    def add(self, chunks: List[Chunk]) -> None:  # type: ignore[override]
        """Embed via the cache; otherwise identical to ChunkIndex.add()."""
        if not chunks:
            return
        new = embed_texts_cached([c.text for c in chunks], self.cache)
        new = l2_normalise(new)
        self.chunks.extend(chunks)
        self.vectors = (new if self.vectors.shape[0] == 0
                        else np.vstack([self.vectors, new]))
        s = self.cache.stats()
        print(f"[index] holds {len(self.chunks)} chunks  "
              f"(matrix shape={self.vectors.shape})  "
              f"cache hits={s['hits']} misses={s['misses']} "
              f"hit_rate={s['hit_rate']}")


if __name__ == "__main__":
    print("[demo] populating the cache with three texts...")
    cache = EmbeddingCache()
    sample = [
        "What is RAG?",
        "How do I install the WidgetMax 3000?",
        "What is RAG?",  # duplicate — second call should hit the cache.
    ]
    out = embed_texts_cached(sample, cache)
    print(f"[demo] shape={out.shape}  stats={cache.stats()}")
    print("[demo] cache dir size:", len(cache), "entries")
    print()
    print("[demo] re-running with the same texts — should be all hits:")
    out2 = embed_texts_cached(sample, cache)
    print(f"[demo] shape={out2.shape}  stats={cache.stats()}")
