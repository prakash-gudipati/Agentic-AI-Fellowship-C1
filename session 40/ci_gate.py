"""Session 41 (merged into S40) — the EVAL GATE in CI.

PROD PATTERN: Eval Gate in CI — the eval suite is only a safety net if a red run
actually BLOCKS a merge. This script runs the suite on one variant and exits
non-zero when the gate fails, exactly like a unit-test step in a CI pipeline.
A bad change is stopped before it ships, not discovered by a customer.

Run directly:
    FAKE_LLM=1 python ci_gate.py V2_FIXED   # exits 0
    FAKE_LLM=1 python ci_gate.py V1_BUGGY   # exits 1
"""
from __future__ import annotations

import sys

from agent_variants import make_variants
from eval_runner import run_suite
from regression_dataset import RegressionDataset


def run_gate(variant_name: str, dataset=None) -> int:
    """Run the suite on a named variant; return a CI exit code (0 pass / 1 fail)."""
    ds = dataset if dataset is not None else RegressionDataset.seed()
    agent = make_variants()[variant_name]
    report = run_suite(agent, ds, variant=variant_name)

    status = "PASS" if report.gate_pass else "FAIL"
    print(f"  EVAL GATE: {status} ({report.passed}/{report.total} cases) "
          f"[variant={variant_name}]")
    if not report.gate_pass:
        failed = [r["id"] for r in report.rows if not r["pass"]]
        print(f"    failing cases: {', '.join(failed)}")
    # Non-zero exit on a red gate is what makes CI block the merge.
    return 0 if report.gate_pass else 1


def main() -> None:
    """CLI: `python ci_gate.py [VARIANT]`. Exits with the gate's status code."""
    variant = sys.argv[1] if len(sys.argv) > 1 else "V2_FIXED"
    sys.exit(run_gate(variant))


if __name__ == "__main__":
    main()
