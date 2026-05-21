"""
Session 22 — test_queries.py

The query set used by compare.py to demonstrate when each retriever wins.
Each entry includes:
  - question: the natural-language query
  - expected_source: the filename that SHOULD show up in the top-K
  - filter: the metadata filter the FilteredRetriever should use
  - redundancy_expected: True if pure similarity is likely to return
                         near-duplicates from the same source — a case
                         where MMR is supposed to help

The "redundancy_expected" flag isn't used as a metric; it's a pedagogical
cue. Students read the column and predict which retriever will look better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TestQuery:
    question: str
    expected_source: str
    filter: Optional[Dict[str, Any]] = None
    redundancy_expected: bool = False
    note: str = ""


TEST_QUERIES = [
    TestQuery(
        question="What is RAG and why does it matter?",
        expected_source="intro_to_rag.pdf",
        filter=None,
        redundancy_expected=True,
        note="Likely 3 near-duplicate intro chunks if pure similarity. MMR should diversify.",
    ),
    TestQuery(
        question="How many stages does a RAG pipeline have, and what are they?",
        expected_source="intro_to_rag.pdf",
        filter={"source_type": "paper"},
        redundancy_expected=False,
        note="Filter narrows to the paper. Similarity probably also wins.",
    ),
    TestQuery(
        question="How do I install the WidgetMax 3000?",
        expected_source="product_manual.pdf",
        filter={"source_type": "manual"},
        redundancy_expected=False,
        note="Filter narrows to the manual. Pure similarity might pull in paper chunks.",
    ),
    TestQuery(
        question="What does the front LED of the WidgetMax 3000 mean when it's solid red?",
        expected_source="product_manual.pdf",
        filter={"source_type": "manual"},
        redundancy_expected=False,
        note="Manual-only question. The filter is the right move.",
    ),
    TestQuery(
        question="Why does this article prefer RAG over fine-tuning?",
        expected_source="sample_article.html",
        filter={"source_type": "blog"},
        redundancy_expected=True,
        note="Blog argument is stated multiple times — MMR should pull diverse paragraphs.",
    ),
    TestQuery(
        question="When is fine-tuning still the right choice?",
        expected_source="sample_article.html",
        filter=None,
        redundancy_expected=False,
        note="No filter — see whether similarity alone finds the counter-section.",
    ),
]
