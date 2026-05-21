"""
Session 23 - test_queries.py

Eight queries used by compare.py AND matrix.py. The first six are inherited
from S22 verbatim. The last two are new and target the two new retrievers:

  Q7  "WidgetMax 3000 power button"        -> BM25 wins on rare exact token
  Q8  "When should I prefer retrieval..."  -> reranker beats single-pass
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TestQuery:
    question: str
    expected_source: str
    filter: Optional[Dict[str, Any]] = None
    redundancy_expected: bool = False
    keyword_expected: bool = False
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
        keyword_expected=True,
        note="Product name WidgetMax is rare -- BM25 wins on the exact token.",
    ),
    TestQuery(
        question="What does the front LED of the WidgetMax 3000 mean when it's solid red?",
        expected_source="product_manual.pdf",
        filter={"source_type": "manual"},
        redundancy_expected=False,
        keyword_expected=True,
        note="Manual-only question. Filter + BM25 both win cleanly.",
    ),
    TestQuery(
        question="Why does this article prefer RAG over fine-tuning?",
        expected_source="sample_article.html",
        filter={"source_type": "blog"},
        redundancy_expected=True,
        note="Blog argument is stated multiple times -- MMR / rerank should pull diverse paragraphs.",
    ),
    TestQuery(
        question="When is fine-tuning still the right choice?",
        expected_source="sample_article.html",
        filter=None,
        redundancy_expected=False,
        note="No filter -- see whether similarity alone finds the counter-section.",
    ),

    # ----- S23 additions: exercise BM25 + rerank -------------------------------
    TestQuery(
        question="WidgetMax 3000 power button",
        expected_source="product_manual.pdf",
        filter=None,
        redundancy_expected=False,
        keyword_expected=True,
        note="Product code + specific component name -- pure cosine washes the exact tokens.",
    ),
    TestQuery(
        question="When should I prefer retrieval over training a smaller model on my data?",
        expected_source="sample_article.html",
        filter=None,
        redundancy_expected=False,
        note="Paraphrased intent. Rerank (LLM-as-judge) should outperform single-pass on this.",
    ),
]
