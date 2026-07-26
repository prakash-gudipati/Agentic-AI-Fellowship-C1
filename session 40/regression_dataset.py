"""Session 41 (merged into S40) — the REGRESSION DATASET.

PROD PATTERN: Regression Dataset — every escaped bug becomes a PERMANENT eval
case. A golden dataset is your starting yardstick; the regression dataset is how
that yardstick GROWS. Once a bug has its own case, it can never silently come
back — the next change that reintroduces it turns the gate red.

This wraps a growing list of EvalCase, seeded from the GOLDEN cases plus any
"previously escaped bug" cases stored alongside in regression_cases.json. It can
load/save to JSON so the dataset is a durable artifact, not code you have to edit.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List, Optional

from eval_dataset import EvalCase, GOLDEN

_HERE = Path(__file__).resolve().parent
_DEFAULT_CASES = _HERE / "regression_cases.json"


class RegressionDataset:
    """A growing, JSON-backed collection of EvalCase objects."""

    def __init__(self, cases: Optional[List[EvalCase]] = None):
        self.cases: List[EvalCase] = list(cases) if cases else []

    # --- iteration so the harness can treat it exactly like GOLDEN ----------
    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def ids(self) -> List[str]:
        """The case ids currently covered — handy for 'is this bug covered?'."""
        return [c.id for c in self.cases]

    # --- the EDD move: a found bug becomes a permanent case -----------------
    def add_from_failure(self, case: EvalCase) -> bool:
        """PROD PATTERN: turn a found bug into a permanent case. Idempotent by id.

        Returns True if the case was newly added, False if its id already exists
        (so re-running a demo doesn't duplicate the same regression case).
        """
        if any(c.id == case.id for c in self.cases):
            return False
        self.cases.append(case)
        return True

    # --- durability: the dataset is an artifact, not source code ------------
    def save(self, path) -> None:
        """Write the whole dataset to JSON so it survives across runs/processes."""
        rows = [asdict(c) for c in self.cases]
        Path(path).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "RegressionDataset":
        """Read a dataset back from a JSON file written by save()."""
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([EvalCase(**row) for row in rows])

    # --- constructors -------------------------------------------------------
    @classmethod
    def seed(cls, extra_path=None) -> "RegressionDataset":
        """Seed from the GOLDEN cases PLUS the stored escaped-bug cases.

        This is the realistic starting point: your existing golden suite, with the
        regression cases for every bug that has previously escaped to production.
        """
        ds = cls(list(GOLDEN))
        path = Path(extra_path) if extra_path else _DEFAULT_CASES
        if path.exists():
            for row in json.loads(path.read_text(encoding="utf-8")):
                ds.add_from_failure(EvalCase(**row))
        return ds
