"""
Session 25 — evals/golden_dataset.py

Loader for the golden dataset. Single source of truth — every harness
reads from here.

A "golden dataset" is the list of (question, ground_truth_answer,
ground_truth_contexts) triples we measure the pipeline against. The
quality of the eval is bounded by the quality of this file. If you put
loose or vague answers in here, no framework can recover.

Production pattern reinforced:
  - structured data lives in JSON, not in Python literals
    (so it can be edited, diffed, and reviewed in isolation)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(_HERE, "golden_dataset.json")


@dataclass
class GoldenRow:
    """One eval row — a question, the answer we expect, and the contexts
    that should support that answer."""
    question: str
    ground_truth: str
    ground_truth_contexts: List[str]
    metadata: Dict[str, Any]

    def as_ragas_dict(self) -> Dict[str, Any]:
        """Shape Ragas's evaluate() function expects per row."""
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "reference_contexts": self.ground_truth_contexts,
        }


def load_golden_dataset(path: Optional[str] = None) -> List[GoldenRow]:
    """Load the 10-row golden dataset (or any compatible JSON)."""
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"golden dataset not found at {path}. "
            "Did you remove evals/golden_dataset.json?"
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: List[GoldenRow] = []
    for r in payload.get("rows", []):
        rows.append(GoldenRow(
            question=r["question"],
            ground_truth=r["ground_truth"],
            ground_truth_contexts=r.get("ground_truth_contexts", []),
            metadata=r.get("metadata", {}),
        ))
    print(f"[golden] loaded {len(rows)} eval rows from {os.path.basename(path)}")
    return rows


if __name__ == "__main__":
    rows = load_golden_dataset()
    for i, row in enumerate(rows[:3], start=1):
        print(f"\n--- Row {i} ---")
        print(f"Q: {row.question}")
        print(f"A: {row.ground_truth[:100]}...")
        print(f"meta: {row.metadata}")
