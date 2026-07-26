"""
Session 33 — retrieval_tools.py

Wraps the Chroma collection behind two named tools the agent can call:

  search_kb(query, k=4, source_filter=None) -> List[RetrievedChunk]
      The main retrieval tool. Embeds the query, runs a top-k search,
      returns typed chunks. Optional `source_filter` lets the agent
      narrow to one document when it knows the right file.

  list_documents() -> List[str]
      Returns the filenames in the corpus. Cheap. The agent uses this
      when the user names a topic that maps clearly to a document
      (e.g. "pricing" → 02_pricing.md). Avoids a wasted retrieval.

PROD PATTERN: Retrieval as a Tool
  The vector store is hidden behind a named tool with a sharp
  description. Other parts of the codebase NEVER call Chroma directly.
  This means we can swap the backing store (Pinecone, Weaviate, BM25)
  by changing one file, and the agent's prompts and tool schemas stay
  intact. Same discipline as S30's tool-design lesson, applied to
  retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from embeddings import embed
from rag_types import RetrievedChunk


# ----------------------------------------------------------------------------
# Lazy collection handle — opened once per process
# ----------------------------------------------------------------------------


_COLL = None  # type: ignore


def _get_collection():
    global _COLL
    if _COLL is None:
        from ingest import _get_chroma_collection

        _COLL = _get_chroma_collection(persist=True)
    return _COLL


def reset_collection_handle() -> None:
    """Force the next call to re-open the collection. Used by tests."""

    global _COLL
    _COLL = None


# ----------------------------------------------------------------------------
# search_kb — the main retrieval tool
# ----------------------------------------------------------------------------


def search_kb(
    query: str,
    k: int = 4,
    source_filter: Optional[str] = None,
) -> List[RetrievedChunk]:
    """Top-k retrieval against the PrepDeck KB.

    The agent calls this through the tool dispatcher in tools.py. It
    NEVER calls this function directly from prompts — only via the
    function-calling interface, so the trace logger sees every call.
    """

    q_vec = embed(query)

    coll = _get_collection()
    kwargs: Dict[str, Any] = {
        "query_embeddings": [q_vec],
        "n_results": max(1, min(k, 20)),
    }
    if source_filter:
        kwargs["where"] = {"source": source_filter}

    res = coll.query(**kwargs)

    out: List[RetrievedChunk] = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for i, doc, meta, dist in zip(ids, docs, metas, dists):
        out.append(
            RetrievedChunk(
                chunk_id=i,
                text=doc,
                source=(meta or {}).get("source", "?"),
                distance=float(dist),
            )
        )
    return out


# ----------------------------------------------------------------------------
# list_documents — cheap helper the agent can call instead of a real search
# ----------------------------------------------------------------------------


def list_documents() -> List[str]:
    """Return the unique filenames stored in the collection."""

    coll = _get_collection()
    snap = coll.get()
    seen = set()
    out: List[str] = []
    for meta in snap.get("metadatas", []) or []:
        if not meta:
            continue
        src = meta.get("source")
        if src and src not in seen:
            seen.add(src)
            out.append(src)
    return sorted(out)


# ----------------------------------------------------------------------------
# Tool schemas — Anthropic-style. (Mirrors what S30 introduced.)
# ----------------------------------------------------------------------------


SEARCH_KB_SCHEMA: Dict[str, Any] = {
    "name": "search_kb",
    # The description is what the model reads to decide WHEN to call this.
    # Notice: we tell it explicitly when NOT to call it.
    "description": (
        "Search the PrepDeck knowledge base for chunks of text relevant to "
        "a specific question. Use this for ANY factual question about "
        "PrepDeck — pricing, refund policy, engineering team, AI stack, "
        "product roadmap, hiring process, on-call rotation, etc. "
        "Do NOT call this for arithmetic, general knowledge questions, "
        "or follow-up questions you can answer from chunks already "
        "retrieved in this turn. Returns up to k chunks ranked by "
        "semantic similarity."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The retrieval query. Should be a focused question or "
                    "phrase, NOT the user's raw multi-part question. If the "
                    "user asks two things, call this tool twice with two "
                    "different queries."
                ),
            },
            "k": {
                "type": "integer",
                "description": (
                    "Number of chunks to retrieve (1-10). Default 4. Use 6-8 "
                    "for broad topics, 2-3 when you already know the source."
                ),
            },
            "source_filter": {
                "type": "string",
                "description": (
                    "Optional. Restrict the search to one document filename "
                    "(e.g. '02_pricing.md'). Only set when you are confident "
                    "the answer lives in that specific document."
                ),
            },
        },
        "required": ["query"],
    },
}


LIST_DOCUMENTS_SCHEMA: Dict[str, Any] = {
    "name": "list_documents",
    "description": (
        "Return the filenames of every document in the PrepDeck knowledge "
        "base. Useful when you need to know the SHAPE of the corpus before "
        "deciding which retrieval queries to issue. Cheap — no embeddings. "
        "Do not call more than once per turn."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


TOOL_SCHEMAS: List[Dict[str, Any]] = [SEARCH_KB_SCHEMA, LIST_DOCUMENTS_SCHEMA]
