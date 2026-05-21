"""
Session 25 — evals/score_log.py

THE PATTERN — every eval run gets a row in a CSV.

The CSV is the regression log. Looking at it across runs is how you spot
"the change yesterday dropped faithfulness from 0.86 to 0.71" before
users do.

Each row carries:
  - UTC timestamp           (when the run happened)
  - git commit hash         (which code produced it)
  - retriever_name          (e.g. "similarity", "metadata_filtered")
  - cache_hit_rate          (the embedding cache stats for this run)
  - all 5 metric scores     (the 4 Ragas metrics + DeepEval)
  - notes                   (free-form, e.g. "added contextual prefixes")

We do NOT use pandas here on purpose — the standard library `csv` module
is enough and keeps requirements light.

Production patterns introduced (S25):
  - score-with-git-hash      (NEW, named today)
  - append-only score log    (NEW, named today)
  - timestamp every artifact (reinforced)
"""

from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.path.join(_HERE, "score_log.csv")

FIELDS = [
    "timestamp_utc",
    "git_hash",
    "retriever_name",
    "cache_hit_rate",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "intent_faithfulness",
    "custom_pass_rate",
    "notes",
]


def _git_hash() -> str:
    """Best-effort `git rev-parse --short HEAD`. Returns '' if not a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_HERE,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def append_run(row: Dict[str, Any],
               path: Optional[str] = None) -> str:
    """Append one row to the score log. Returns the path that was written.

    Missing keys default to '' for strings and 0.0 for numbers — so callers
    can pass partial dicts without crashing.
    """
    path = path or DEFAULT_LOG
    new_file = not os.path.exists(path)

    full_row: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash":      _git_hash(),
        "retriever_name": "",
        "cache_hit_rate": 0.0,
        "faithfulness":   0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "intent_faithfulness": 0.0,
        "custom_pass_rate": 0.0,
        "notes": "",
    }
    full_row.update({k: v for k, v in row.items() if k in FIELDS})

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(full_row)

    print(f"[scorelog] appended  {os.path.basename(path)}  "
          f"hash={full_row['git_hash']!r}  "
          f"F={full_row['faithfulness']:.2f}  "
          f"AR={full_row['answer_relevancy']:.2f}  "
          f"CP={full_row['context_precision']:.2f}  "
          f"CR={full_row['context_recall']:.2f}  "
          f"DE={full_row['intent_faithfulness']:.2f}")
    return path


def read_runs(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return every row in the log as a list of dicts. Empty list if missing."""
    path = path or DEFAULT_LOG
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    append_run({
        "retriever_name": "demo",
        "faithfulness": 0.84,
        "answer_relevancy": 0.79,
        "context_precision": 0.71,
        "context_recall": 0.62,
        "intent_faithfulness": 0.66,
        "custom_pass_rate": 0.60,
        "notes": "smoke test row",
    })
    print("\nAll runs in the log:")
    for row in read_runs():
        print(" ", row)
