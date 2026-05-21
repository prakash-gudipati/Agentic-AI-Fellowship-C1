"""
Session 24 - patterns/parent_child.py

PATTERN 1 - Parent-child retrieval.  "Small to match, big to read."

The conflict every chunker faces:
  - SMALL chunks (one sentence) match queries precisely - cosine actually
    fires on the right concept, not a wider blob.
  - LARGE chunks (a paragraph or a section) give the LLM enough surrounding
    context to actually answer.

You cannot have both with a single chunk size. Parent-child solves it:

  1. At INDEX time, chunk each document twice:
       parents = large chunks (one per section, e.g. 1500-2000 chars)
       children = small chunks of each parent (e.g. 200-400 chars)
     Each child remembers its parent_id.

  2. EMBED the children only. The children are what get matched.

  3. At RETRIEVAL time:
       - score children against the query (precise match)
       - look up each winning child's parent_id
       - de-duplicate (multiple children may point at the same parent)
       - return the PARENTS to the LLM

That is the whole pattern. We trade index size (slightly bigger - we store
parents AND children) for retrieval quality.

Production patterns reinforced (S24):
  - retriever strategy pattern (S22)
  - INDEX-TIME INVESTMENT FOR QUERY-TIME GAINS (named pattern, NEW today)
  - compose, don't replace (named pattern, NEW today)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

# Resolve sibling modules when imported as patterns.parent_child.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from chunker import Chunk
from loaders import Doc
from retrievers.base import Retriever, Retrieved, ChunkIndex
from retrievers.similarity import SimilarityRetriever


# ----- Defaults ---------------------------------------------------------------
PARENT_SIZE = 1600
PARENT_OVERLAP = 0
CHILD_SIZE = 350
CHILD_OVERLAP = 50


# ----- Index ------------------------------------------------------------------

@dataclass
class _ParentBlock:
    """A large block of text returned to the LLM."""
    parent_id: int
    source: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParentChildIndex:
    """Holds parents (returned) and a ChunkIndex over their children (matched).

    Build with:
        idx = ParentChildIndex()
        idx.build(docs)

    Then hand it to ParentChildRetriever like any other index.
    """

    def __init__(self) -> None:
        self.parents: List[_ParentBlock] = []
        # _child_to_parent[child_index_in_chunk_index] = parent_id
        self._child_to_parent: List[int] = []
        # We reuse the S22 ChunkIndex for embedded children so cosine math
        # is identical to similarity-based retrieval.
        self.child_index = ChunkIndex()

    def build(self, docs: List[Doc]) -> None:
        """Chunk each doc into large parents and small children. Embed children."""
        child_chunks: List[Chunk] = []
        next_parent = 0
        next_child = 0

        for d in docs:
            # PARENTS - simple character window. Production: respect headers.
            parents_of_d = _window(d.text, PARENT_SIZE, PARENT_OVERLAP)
            for ptext in parents_of_d:
                p = _ParentBlock(parent_id=next_parent, source=d.source,
                                 text=ptext, metadata=dict(d.metadata))
                self.parents.append(p)

                # CHILDREN of this parent - smaller window.
                children_of_p = _window(ptext, CHILD_SIZE, CHILD_OVERLAP)
                for c_text in children_of_p:
                    c = Chunk(chunk_id=next_child, source=d.source,
                              text=c_text, metadata=dict(d.metadata))
                    child_chunks.append(c)
                    self._child_to_parent.append(next_parent)
                    next_child += 1
                next_parent += 1
            print(f"[parent_child] {d.source:<24} -> "
                  f"{len(parents_of_d):>3} parents")

        print(f"[parent_child] total: {len(self.parents)} parents, "
              f"{len(child_chunks)} children")
        self.child_index.add(child_chunks)

    def parent_of(self, child_idx: int) -> _ParentBlock:
        """Look up the parent block of a given child position in child_index."""
        return self.parents[self._child_to_parent[child_idx]]


def _window(text: str, size: int, overlap: int) -> List[str]:
    """Plain character window. Used for both parents and children."""
    text = text.strip()
    if not text:
        return []
    step = max(size - overlap, 1)
    pieces: List[str] = []
    for i in range(0, len(text), step):
        chunk = text[i:i + size].strip()
        if chunk:
            pieces.append(chunk)
    return pieces


# ----- Retriever --------------------------------------------------------------

class ParentChildRetriever:
    """Conforms to the retriever interface but reads from a ParentChildIndex.

    NOT a subclass of Retriever - the ABC expects a ChunkIndex, and we need
    the richer ParentChildIndex. The public surface (`pick(query, k)`) is
    identical, which is what matters for the strategy pattern.
    """

    name = "parent_child"

    def __init__(self, index: ParentChildIndex) -> None:
        self.index = index
        # Reuse SimilarityRetriever on the child ChunkIndex - cosine math
        # for free. Could swap for BM25 or hybrid; that's the strategy
        # pattern doing its job.
        self._inner = SimilarityRetriever(index.child_index)

    def pick(self, query: str, k: int = 3, fetch_k: int = 10,
             **kwargs: Any) -> List[Retrieved]:
        """Match against children, return de-duplicated parents."""
        if len(self.index.child_index) == 0:
            return []

        child_hits = self._inner.pick(query, k=fetch_k)
        if not child_hits:
            return []

        # De-duplicate parents in order of best child score.
        seen: set = set()
        out: List[Retrieved] = []
        for hit in child_hits:
            child_idx = self.index.child_index.chunks.index(hit.chunk)
            parent = self.index.parent_of(child_idx)
            if parent.parent_id in seen:
                continue
            seen.add(parent.parent_id)
            # Wrap the parent text in a Chunk so the downstream contract
            # stays identical to every other retriever.
            wrapped = Chunk(chunk_id=parent.parent_id,
                            source=parent.source,
                            text=parent.text,
                            metadata=parent.metadata)
            out.append(Retrieved(chunk=wrapped, score=hit.score))
            if len(out) >= k:
                break

        print(f"[parent_child] query={query!r}  fetch_k={fetch_k}  "
              f"-> {len(out)} parents")
        return out


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, HERE)
    from loaders import load_corpus

    files = [
        os.path.join(HERE, "data", "intro_to_rag.pdf"),
        os.path.join(HERE, "data", "product_manual.pdf"),
        os.path.join(HERE, "data", "sample_article.html"),
    ]
    docs = load_corpus(files)
    idx = ParentChildIndex()
    idx.build(docs)
    r = ParentChildRetriever(idx)
    q = " ".join(sys.argv[1:]) or "What is RAG?"
    print()
    for hit in r.pick(q, k=2):
        preview = hit.chunk.text[:120].replace("\n", " ")
        print(f"  score={hit.score:.3f}  src={hit.chunk.source}")
        print(f"    {preview}...")
