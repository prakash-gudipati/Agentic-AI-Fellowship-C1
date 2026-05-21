"""
Session 21 — test_queries.py

The ground-truth query set for the chunking benchmark. Each entry pairs a
question with the source filename whose chunks SHOULD be retrieved.

A retrieval is a "hit" when at least one of the top-K chunks comes from the
expected source. This is the simplest possible retrieval-quality metric — a
warm-up for the proper evaluation tooling we add in S25 (Ragas + DeepEval).
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TestQuery:
    question: str
    expected_source: str  # filename — must match Chunk.source exactly
    note: str = ""        # human-readable label for the benchmark output


TEST_QUERIES: List[TestQuery] = [
    TestQuery(
        question="What is RAG and why does it matter?",
        expected_source="intro_to_rag.pdf",
        note="paper · headline definition",
    ),
    TestQuery(
        question="How many stages does a RAG pipeline have, and what are they?",
        expected_source="intro_to_rag.pdf",
        note="paper · 5-stage list",
    ),
    TestQuery(
        question="How do I install the WidgetMax 3000?",
        expected_source="product_manual.pdf",
        note="manual · installation",
    ),
    TestQuery(
        question="What does the front LED of the WidgetMax 3000 mean when it's solid red?",
        expected_source="product_manual.pdf",
        note="manual · troubleshooting",
    ),
    TestQuery(
        question="Why does this article prefer RAG over fine-tuning?",
        expected_source="sample_article.html",
        note="blog · core argument",
    ),
    TestQuery(
        question="When is fine-tuning still the right choice?",
        expected_source="sample_article.html",
        note="blog · counter-section",
    ),
]
