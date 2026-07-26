"""
Session 41 — Fine-Tuning Fundamentals: When and How
Module 7 of 7: demo.py

Five demo modes (run with FAKE_LLM=1 for offline use):

  python demo.py prepare   — build + validate training JSONL (Step 1)
  python demo.py upload    — upload file, get file_id (Step 2, fake in FAKE_LLM mode)
  python demo.py monitor   — poll job, print loss curves + overfitting detection (Step 4)
  python demo.py eval      — compare base vs fine-tuned, apply eval gate (Step 5)
  python demo.py versions  — show version registry + prompt lock demo
  python demo.py all       — run all five in sequence (for smoke-testing)
"""

import os
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; set env vars manually if needed

FAKE = os.environ.get("FAKE_LLM", "0") == "1"


def run_prepare():
    print("=" * 60)
    print("DEMO 1 — Prepare training data")
    print("=" * 60)
    from prepare_data import main as prepare_main
    train_path, eval_path, prompt_hash = prepare_main()
    print(f"\n[DEMO 1] Done. Files ready for upload.")
    return train_path, eval_path


def run_upload(train_path=None):
    print("\n" + "=" * 60)
    print("DEMO 2 — Upload training file")
    print("=" * 60)
    from upload_file import main as upload_main
    file_id = upload_main(str(train_path) if train_path else "training_data.jsonl")
    print(f"\n[DEMO 2] file_id saved. Ready to start job.")
    return file_id


def run_monitor():
    print("\n" + "=" * 60)
    print("DEMO 3 — Monitor training job (loss curves + overfitting)")
    print("=" * 60)
    from start_job import main as start_main
    from monitor_job import main as monitor_main
    if FAKE:
        # Ensure a fake job_id exists
        start_main()
    model_id = monitor_main()
    print(f"\n[DEMO 3] Monitoring complete.")
    return model_id


def run_eval():
    print("\n" + "=" * 60)
    print("DEMO 4 — Eval gate: base model vs fine-tuned model")
    print("=" * 60)
    from run_eval import main as eval_main
    passed = eval_main()
    status = "GATE PASS ✓" if passed else "GATE FAIL ✗"
    print(f"\n[DEMO 4] Eval complete — {status}")
    return passed


def run_versions():
    print("\n" + "=" * 60)
    print("DEMO 5 — Version registry + Prompt–Model Lock")
    print("=" * 60)
    from version_config import main as version_main
    version_main()
    print(f"\n[DEMO 5] Version registry complete.")


def run_all():
    train_path, _ = run_prepare()
    run_upload(train_path)
    run_monitor()
    run_eval()
    run_versions()
    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE")
    print("=" * 60)
    if FAKE:
        print("[OK] All 5 demo modes passed in FAKE_LLM=1 mode.")


MODES = {
    "prepare":  run_prepare,
    "upload":   run_upload,
    "monitor":  run_monitor,
    "eval":     run_eval,
    "versions": run_versions,
    "all":      run_all,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print("Usage: python demo.py <mode>")
        print(f"Modes: {', '.join(MODES.keys())}")
        print()
        print("Offline (no API key needed):")
        print("  FAKE_LLM=1 python demo.py all")
        sys.exit(1)

    mode = sys.argv[1]
    if FAKE:
        print(f"[DEMO] Running in FAKE_LLM=1 mode (offline, no API calls)")
    MODES[mode]()


if __name__ == "__main__":
    main()
