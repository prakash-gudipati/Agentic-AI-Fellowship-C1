"""Session 41 (merged into S40) — Eval-Driven Development demos.

EDD = once you HAVE an eval harness, you use it as your development LOOP, like
TDD but for AI answer quality. Four runnable demos, all fully OFFLINE:

  python edd_demo.py [1|2|3|4|all]
  FAKE_LLM=1 python edd_demo.py all

  1  Eval-First (red->green): V1_BUGGY fails the gate; V2_FIXED makes it green.
  2  Regression Dataset: the escaped bug becomes a permanent, saved case.
  3  A/B Experiment: same dataset, two variants, pick the winner by score.
  4  Eval Gate in CI: green variant exits 0, regressed variant exits 1.
"""
from __future__ import annotations

import os
import sys
import tempfile

# Optional .env loader — harmless if python-dotenv isn't installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import trace_logger
from agent_variants import make_variants, BUGGY_QUESTION
from regression_dataset import RegressionDataset
from eval_runner import run_suite, save_run, load_run, print_run
from experiment import run_experiment
from ci_gate import run_gate
from eval_dataset import EvalCase, GOLDEN, _KB


# --- Demo 1: Eval-First Development (red -> green) ---------------------------

def demo1():
    trace_logger.banner("EDD DEMO 1 - Eval-First Development (red -> green)")
    print("  Discipline: write the FAILING case first; the fix is done when it goes green.\n")
    dataset = RegressionDataset(list(GOLDEN))  # the existing golden suite
    variants = make_variants()

    print("  --- run the suite on V1_BUGGY (the regressed prompt) ---")
    red = run_suite(variants["V1_BUGGY"], dataset, variant="V1_BUGGY")
    print_run(red)
    failed = [r["id"] for r in red.rows if not r["pass"]]
    print(f"  RED: gate {'PASS' if red.gate_pass else 'FAIL'} - failing case(s): {failed}\n")

    print("  --- apply the fix: switch to V2_FIXED, rerun the SAME suite ---")
    green = run_suite(variants["V2_FIXED"], dataset, variant="V2_FIXED")
    print_run(green)
    print(f"  GREEN: gate {'PASS' if green.gate_pass else 'FAIL'}")
    print("\n  We wrote the failing case first; the fix is what turns the gate green.")


# --- Demo 2: Regression Dataset (a bug becomes a permanent case) -------------

def demo2():
    trace_logger.banner("EDD DEMO 2 - Regression Dataset (escaped bug -> permanent case)")
    print("  PROD PATTERN: every escaped bug becomes a permanent eval case.\n")

    dataset = RegressionDataset(list(GOLDEN))
    print(f"  golden cases at start: {len(dataset)}  ({', '.join(dataset.ids())})\n")

    # The bug from demo 1: the settlement question that V1 hallucinated on.
    bug_case = EvalCase(
        id="reg_transfers_settle",
        question=BUGGY_QUESTION,
        context=_KB,
        label="REGRESSION: v1 hallucinated '7 business days' here - lock the grounded answer",
        must_block=False,
    )
    added = dataset.add_from_failure(bug_case)
    print(f"  add_from_failure -> newly added: {added}   dataset now: {len(dataset)} cases")

    # Persist + reload so it survives across runs/processes (it's an artifact).
    path = os.path.join(tempfile.gettempdir(), "s40_regression.json")
    dataset.save(path)
    reloaded = RegressionDataset.load(path)
    print(f"  saved -> {path}")
    print(f"  reloaded from disk: {len(reloaded)} cases, bug covered = "
          f"{'reg_transfers_settle' in reloaded.ids()}\n")

    print("  --- rerun V2_FIXED against the GROWN dataset ---")
    report = run_suite(make_variants()["V2_FIXED"], reloaded, variant="V2_FIXED")
    print_run(report)
    print("\n  The once-broken case is now permanently covered - and still green.")


# --- Demo 3: A/B Experiment --------------------------------------------------

def demo3():
    trace_logger.banner("EDD DEMO 3 - A/B Experiment (prompt v1 vs prompt v2)")
    print("  PROD PATTERN: run BOTH variants on the same dataset; pick the winner by score.\n")
    dataset = RegressionDataset.seed()  # golden + the stored escaped-bug cases
    print(f"  experiment dataset: {len(dataset)} cases\n")

    variants = make_variants()
    result = run_experiment(
        dataset,
        {"prompt_v1": variants["V1_BUGGY"], "prompt_v2": variants["V2_FIXED"]},
    )
    print(f"\n  Experiment winner: {result.winner}")


# --- Demo 4: Eval Gate in CI -------------------------------------------------

def demo4():
    trace_logger.banner("EDD DEMO 4 - Eval Gate in CI (exit 0 vs exit 1)")
    print("  PROD PATTERN: the gate exits non-zero on a regression, blocking the merge.\n")

    print("  --- good change: V2_FIXED ---")
    code_ok = run_gate("V2_FIXED")
    print(f"  -> ci exit code: {code_ok}  ({'merge allowed' if code_ok == 0 else 'merge blocked'})\n")

    print("  --- bad change: V1_BUGGY ---")
    code_bad = run_gate("V1_BUGGY")
    print(f"  -> ci exit code: {code_bad}  ({'merge allowed' if code_bad == 0 else 'merge blocked'})")

    print("\n  Same gate, two changes: the regression is stopped before it ships.")


DEMOS = {"1": demo1, "2": demo2, "3": demo3, "4": demo4}


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        for key in ["1", "2", "3", "4"]:
            DEMOS[key]()
            print()
    elif arg in DEMOS:
        DEMOS[arg]()
    else:
        print(f"unknown demo {arg!r}; use 1|2|3|4|all")
        sys.exit(2)


if __name__ == "__main__":
    main()
