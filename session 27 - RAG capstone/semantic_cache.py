"""
Session 25 — semantic_cache.py     (the OTHER cache pattern)

Why this file exists
--------------------
embedding_cache.py is an EXACT cache:
    key = (model, sha256(chunk_text))
    same text → same hash → cache hit.   different text → guaranteed miss.

That's the right cache for ingest-time work (we're embedding chunks of OUR
own corpus, where text is stable). It's the WRONG cache for serving real
users at query time, because real users don't type the same question
twice in a row — they type CLOSE questions:

    "What time does the store open?"
    "When does the store open?"
    "store opening hours?"

All three deserve the same cached answer. An exact cache would miss all
three. A SEMANTIC cache hits on all three.

Design
------
- Embed the incoming question (same model used everywhere else in S25).
- L2-normalise so cosine == dot product.
- Disk-backed: each cache entry is a JSON file under .semantic_response_cache/
  containing the question text, its embedding vector, the cached answer,
  the retrieved contexts, the cached-at timestamp, and the source TTL.
- Lookup: linear scan with cosine similarity, return the first entry whose
  cosine to the query is above `threshold` (default 0.95).
- For real production scale you would put these vectors in a real vector
  DB (S19) — same algorithm, different storage. We keep a linear scan
  for teaching clarity.

When to reach for this
----------------------
- Serving real user queries (Phase 6 shipping)
- Customer support / FAQ bots
- Any agent layer where the user reissues paraphrases of the same query

When NOT to use it
------------------
- Eval harnesses (you want fresh outputs to score)
- Anywhere correctness depends on the precise wording of the prompt
- When the same wording must produce different answers in different
  sessions (e.g. personalised replies)

Risks called out in the docstring of `lookup()`:
- False positives — "is the store open NOW?" and "when does the store
  open?" are similar embeddings with different right answers.
- Stale answers — the cache is time-blind. We expose `max_age_seconds`
  per lookup so you can scope to "the last hour" if your data ages.

Production patterns introduced (S25):
  - semantic cache by query embedding   (NEW, named today as a contrast
                                          to the exact cache from
                                          embedding_cache.py)
  - configurable similarity threshold   (the wrong threshold can ruin
                                          either precision or recall)
  - cosine via L2-normalised dot product (reused from S20+)
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from embeddings import EMBEDDING_MODEL, embed_one, l2_normalise


DEFAULT_CACHE_DIR = os.path.join(_HERE, ".semantic_response_cache")
DEFAULT_THRESHOLD = 0.95


@dataclass
class SemanticHit:
    """Result of a successful semantic-cache lookup."""
    question: str          # the original question that produced the entry
    answer: str
    contexts: List[str]
    similarity: float      # cosine to the incoming query
    age_seconds: float     # seconds since this entry was written
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticResponseCache:
    """Disk-backed semantic cache of (query_embedding) → answer.

    Each entry on disk is a tiny JSON file:
        {
          "model":       "<embedding model name>",
          "question":    "<original question text>",
          "vector":      [<1536 floats>],
          "answer":      "<cached answer text>",
          "contexts":    [<chunk texts the LLM saw>],
          "saved_at":    <unix timestamp>,
          "metadata":    {<arbitrary extra fields>}
        }

    The model name partitions the cache, so swapping embedding models does
    NOT silently mix old vectors with new ones — same rule as the exact
    cache.
    """

    def __init__(self,
                 cache_dir: str = DEFAULT_CACHE_DIR,
                 model: str = EMBEDDING_MODEL,
                 threshold: float = DEFAULT_THRESHOLD) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0,1], got {threshold!r}")
        self.cache_dir = cache_dir
        self.model = model
        self.model_dir = os.path.join(cache_dir, _safe_filename(model))
        os.makedirs(self.model_dir, exist_ok=True)
        self.threshold = threshold

        # Counters surfaced to the demo so the cache's value is visible.
        self.hits = 0
        self.misses = 0

        # Lazy in-memory index of (path, vector, payload). Populated the first
        # time we run a lookup, kept fresh across put() calls.
        self._entries: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
        self._loaded = False

    # ── Core operations ────────────────────────────────────────────────────

    def lookup(self,
               question: str,
               *,
               threshold: Optional[float] = None,
               max_age_seconds: Optional[float] = None
               ) -> Optional[SemanticHit]:
        """Search the cache for an entry whose query-embedding has cosine
        ≥ threshold with `question`.

        Returns the highest-similarity match, or None if nothing meets
        the bar. If `max_age_seconds` is given, entries older than that
        are ignored (production cache freshness lever).
        """
        thr = threshold if threshold is not None else self.threshold
        self._ensure_loaded()
        if not self._entries:
            self.misses += 1
            return None

        q_vec = l2_normalise(embed_one(question))

        # Stack all vectors → one matmul. Linear-scan is fine at teaching
        # scale; switch to a vector DB (S19) for production scale.
        matrix = np.stack([v for _, v, _ in self._entries])
        scores = matrix @ q_vec
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < thr:
            self.misses += 1
            return None

        _, _, payload = self._entries[best_idx]
        age = time.time() - float(payload.get("saved_at", 0.0))
        if max_age_seconds is not None and age > max_age_seconds:
            # Best match is too old. We treat this as a miss rather than
            # falling through to next-best — staleness is a binary signal.
            self.misses += 1
            return None

        self.hits += 1
        return SemanticHit(
            question=payload.get("question", ""),
            answer=payload.get("answer", ""),
            contexts=payload.get("contexts", []) or [],
            similarity=best_score,
            age_seconds=age,
            metadata=payload.get("metadata", {}) or {},
        )

    def put(self,
            question: str,
            answer: str,
            contexts: List[str],
            *,
            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Insert a (question, answer, contexts) triple. Returns the entry id.

        We embed the QUESTION (not the answer) — the lookup key is what
        the user types, not what the system produces.
        """
        vec = l2_normalise(embed_one(question)).astype(np.float32)
        entry_id = uuid.uuid4().hex
        payload = {
            "model":     self.model,
            "question":  question,
            "vector":    vec.tolist(),
            "answer":    answer,
            "contexts":  contexts,
            "saved_at":  time.time(),
            "metadata":  metadata or {},
        }

        path = os.path.join(self.model_dir, f"{entry_id}.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except Exception as e:
            print(f"[semantic-cache] WARN: failed to write {path} ({e!s})")
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass
            return ""

        # Update the in-memory index so subsequent lookups see this entry
        # without re-scanning the directory.
        self._entries.append((path, vec, payload))
        return entry_id

    # ── Convenience ────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        rate = (self.hits / total) if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_queries": total,
            "hit_rate": round(rate, 3),
            "entries_on_disk": len(self._entries) if self._loaded
                               else self._count_on_disk(),
            "threshold": self.threshold,
            "model": self.model,
        }

    def __len__(self) -> int:
        if self._loaded:
            return len(self._entries)
        return self._count_on_disk()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        for name in sorted(os.listdir(self.model_dir)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.model_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                vec = np.asarray(payload["vector"], dtype=np.float32)
            except Exception as e:
                print(f"[semantic-cache] WARN: skipping {path} ({e!s})")
                continue
            self._entries.append((path, vec, payload))
        self._loaded = True
        print(f"[semantic-cache] loaded {len(self._entries)} entries from "
              f"{self.model_dir}")

    def _count_on_disk(self) -> int:
        try:
            return sum(1 for n in os.listdir(self.model_dir)
                       if n.endswith(".json"))
        except FileNotFoundError:
            return 0


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


# ── The wrapper that turns a pipeline into a cached pipeline ────────────────

def cached_answer(pipeline: Any,
                  question: str,
                  cache: SemanticResponseCache,
                  *,
                  threshold: Optional[float] = None,
                  max_age_seconds: Optional[float] = None,
                  on_hit_label: str = "[semantic-cache] HIT",
                  on_miss_label: str = "[semantic-cache] MISS"
                  ) -> Tuple[Any, bool]:
    """Run `pipeline.answer(question)` if there's no semantic-cache hit.

    Returns (result, was_cache_hit). `result` is either:
      - a PipelineResult-like object (whatever pipeline.answer returns),
        on a cache miss; the result is also persisted into the cache.
      - a SemanticHit, on a cache hit.

    This is the wrapper Phase 6 (S46) would put in front of a FastAPI
    handler. We do not wire it into demo.py because the eval harness
    explicitly wants fresh outputs every time.
    """
    hit = cache.lookup(question,
                       threshold=threshold,
                       max_age_seconds=max_age_seconds)
    if hit is not None:
        print(f"{on_hit_label}  sim={hit.similarity:.3f}  "
              f"age={hit.age_seconds:.1f}s  q='{hit.question[:60]}'")
        return hit, True

    print(f"{on_miss_label}  q='{question[:60]}'  →  calling pipeline")
    result = pipeline.answer(question)
    # Persist for next time. We tolerate PipelineResult OR plain dicts.
    answer_text = getattr(result, "answer", None) or result.get("answer", "")
    contexts    = (getattr(result, "contexts", None)
                   or result.get("contexts", []) or [])
    cache.put(question, answer_text, list(contexts))
    return result, False


# ── A demo when the file is run directly ────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="S25 semantic cache demo — paraphrased queries hit.")
    parser.add_argument("--threshold", type=float, default=0.92,
                        help="cosine similarity threshold (default 0.92 for "
                             "the demo so paraphrases reliably hit)")
    parser.add_argument("--fresh", action="store_true",
                        help="clear the cache before running")
    args = parser.parse_args()

    if args.fresh and os.path.isdir(DEFAULT_CACHE_DIR):
        import shutil
        shutil.rmtree(DEFAULT_CACHE_DIR)
        print(f"[demo] cleared {DEFAULT_CACHE_DIR}")

    cache = SemanticResponseCache(threshold=args.threshold)

    # Mock pipeline — keeps the demo runnable without an LLM key. Any
    # callable that exposes .answer(question) → object with .answer and
    # .contexts works.
    class MockPipeline:
        def __init__(self) -> None:
            self.calls = 0

        def answer(self, question: str) -> Dict[str, Any]:
            self.calls += 1
            return {
                "question": question,
                "answer":   f"(mock) the store opens at 9am every weekday.",
                "contexts": ["Store hours: Mon-Fri 9-6.",
                             "Weekends 10-4."],
            }

    pipe = MockPipeline()

    # Seed: first call ALWAYS misses.
    queries = [
        "What time does the store open?",
        "When does the store open?",       # paraphrase — should HIT
        "store opening hours?",            # paraphrase — should HIT
        "What is RAG?",                    # unrelated — should MISS
    ]
    for q in queries:
        print()
        result, was_hit = cached_answer(pipe, q, cache,
                                        threshold=args.threshold)
        if was_hit:
            print(f"  cached answer: {result.answer}")
        else:
            print(f"  fresh answer:  {result['answer']}")

    print()
    print("[demo] pipeline.answer called", pipe.calls, "times "
          "(would have been", len(queries), "without the cache)")
    print("[demo] cache stats:", cache.stats())
