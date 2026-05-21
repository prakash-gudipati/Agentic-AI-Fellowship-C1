"""
Session 24 - demo.py

The signature demo. Runs the same ambiguous query through:
  1. basic similarity (the S20 baseline)
  2. contextual retrieval (chunks prefixed with LLM-written context)
  3. HyDE
  4. multi-query
  5. parent-child

Usage:
    python demo.py
    python demo.py --skip-contextual    # contextual prefixing is expensive
    python demo.py --query "your question here"

The contextual path pays a one-time LLM call per chunk at INDEX time -
expect 20-50 seconds the first run. Subsequent queries are free.

This demo deliberately uses a small corpus so all five paths run end-to-end
in under two minutes. Drop your domain corpus into data/ for a real test.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from loaders import load_corpus
from chunker import chunk_corpus
from retrievers.base import ChunkIndex
from retrievers.similarity import SimilarityRetriever
from patterns.parent_child import ParentChildIndex, ParentChildRetriever
from patterns.contextual import ContextualChunker
from patterns.hyde import HyDERetriever
from patterns.multi_query import MultiQueryRetriever


DEFAULT_CORPUS = [
    os.path.join(HERE, "data", "intro_to_rag.pdf"),
    os.path.join(HERE, "data", "product_manual.pdf"),
    os.path.join(HERE, "data", "sample_article.html"),
]

DEFAULT_QUERY = "When should I prefer retrieval over training a smaller model on my data?"


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def show_results(label: str, results, t_elapsed: float) -> None:
    print(f"\n  [{label}]  ({t_elapsed:.1f}s)")
    if not results:
        print("    (no results)")
        return
    for hit in results:
        preview = hit.chunk.text[:90].replace("\n", " ")
        print(f"    score={hit.score:.3f}  src={hit.chunk.source:<24}  {preview}...")


def main() -> int:
    parser = argparse.ArgumentParser(description="S24 advanced RAG demo")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY,
                        help="query to run through every pattern")
    parser.add_argument("--k", type=int, default=3,
                        help="top-K results to print")
    parser.add_argument("--skip-contextual", action="store_true",
                        help="skip contextual retrieval (avoids per-chunk LLM calls)")
    parser.add_argument("--skip-hyde", action="store_true",
                        help="skip HyDE (avoids one LLM call per query)")
    parser.add_argument("--skip-multi-query", action="store_true",
                        help="skip multi-query (avoids one LLM call per query)")
    args = parser.parse_args()

    missing = [p for p in DEFAULT_CORPUS if not os.path.exists(p)]
    if missing:
        print("ERROR: corpus is missing:")
        for p in missing:
            print("  -", p)
        return 1

    # ----- INGEST + base index ------------------------------------------------
    header("INDEX-TIME")
    docs = load_corpus(DEFAULT_CORPUS)
    chunks = chunk_corpus(docs)
    base_index = ChunkIndex()
    base_index.add(chunks)

    # Parent-child needs its own dedicated index.
    print("\n[parent_child] building parent-child index...")
    pc_index = ParentChildIndex()
    pc_index.build(docs)

    # Contextual needs to chunk-then-prefix-then-embed. One LLM call per
    # chunk at INGEST time. Off by default in this demo - too slow for
    # live teaching.
    contextual_index = None
    if not args.skip_contextual:
        print("\n[contextual] building contextual index (LLM calls follow)...")
        c_chunker = ContextualChunker(chunk_corpus)
        c_chunks = c_chunker.chunk(docs)
        contextual_index = ChunkIndex()
        contextual_index.add(c_chunks)

    # ----- QUERY-TIME ---------------------------------------------------------
    header(f"QUERY-TIME  -  {args.query!r}")

    base = SimilarityRetriever(base_index)

    # 1. Plain similarity baseline.
    t0 = time.time()
    results = base.pick(args.query, k=args.k)
    show_results("BASELINE  similarity", results, time.time() - t0)

    # 2. Contextual retrieval (if built).
    if contextual_index is not None:
        c_base = SimilarityRetriever(contextual_index)
        t0 = time.time()
        results = c_base.pick(args.query, k=args.k)
        show_results("CONTEXTUAL  similarity-on-contextualised-chunks",
                     results, time.time() - t0)

    # 3. HyDE.
    if not args.skip_hyde:
        hyde = HyDERetriever(base_index, base=SimilarityRetriever(base_index))
        t0 = time.time()
        results = hyde.pick(args.query, k=args.k)
        show_results("HYDE", results, time.time() - t0)

    # 4. Multi-query.
    if not args.skip_multi_query:
        mq = MultiQueryRetriever(base_index,
                                 base=SimilarityRetriever(base_index),
                                 n_variants=3)
        t0 = time.time()
        results = mq.pick(args.query, k=args.k)
        show_results("MULTI-QUERY  (3 variants, RRF fused)",
                     results, time.time() - t0)

    # 5. Parent-child.
    pc = ParentChildRetriever(pc_index)
    t0 = time.time()
    results = pc.pick(args.query, k=args.k)
    show_results("PARENT-CHILD  (small-match, big-return)",
                 results, time.time() - t0)

    header("READING THE OUTPUT")
    print("  - Look at the sources column - do the right docs come back?")
    print("  - Look at the chunk preview - is it the section that actually answers?")
    print("  - HyDE often surfaces NEW chunks the baseline missed because it")
    print("    searches in answer-space, not question-space.")
    print("  - Contextual retrieval lifts the score of chunks whose meaning")
    print("    only made sense IN context (e.g. 'the rate dropped to 12%').")
    print("  - Parent-child returns larger blocks - good for the LLM, harder")
    print("    to read in this terminal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
