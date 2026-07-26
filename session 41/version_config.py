"""
Session 41 — Fine-Tuning Fundamentals: When and How
Module 6 of 7: version_config.py

PROD PATTERNS:
  Fine-Tuned Model Versioning  — tag every model ID with metadata
  Prompt–Model Version Lock    — store system prompt hash alongside model ID
  Training Data Provenance     — each version entry records dataset version

Never hardcode a model ID in application code.
Always read from FT_MODEL_ID environment variable.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

VERSION_FILE = Path(os.environ.get("FT_VERSION_FILE", "ft_versions.json"))
CONFIG_PATH  = Path(os.environ.get("FT_CONFIG_PATH", "ft_config.json"))


# ── PROMPT HASH ───────────────────────────────────────────────────────────────

def hash_prompt(text: str) -> str:
    """Return first 16 hex chars of SHA-256 of the prompt text."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── ACTIVE MODEL ──────────────────────────────────────────────────────────────

def load_active_model() -> str:
    """
    PROD PATTERN: Model ID comes from env var — NEVER hardcoded.
    Falls back to ft_config.json, then to an empty string.
    """
    from_env = os.environ.get("FT_MODEL_ID", "")
    if from_env:
        return from_env
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
        return cfg.get("fine_tuned_model", "")
    return ""


# ── VERSION REGISTRY ──────────────────────────────────────────────────────────

def load_versions() -> list[dict]:
    if VERSION_FILE.exists():
        return json.loads(VERSION_FILE.read_text())
    return []


def save_versions(versions: list[dict]) -> None:
    VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(json.dumps(versions, indent=2))


def add_version(
    model_id: str,
    dataset_version: str,
    eval_score: float,
    system_prompt: str,
    note: str = "",
) -> dict:
    """
    Record a new fine-tuned model version.
    PROD PATTERN: Prompt–Model Version Lock — prompt hash stored alongside model ID.
    """
    entry = {
        "model_id":        model_id,
        "dataset_version": dataset_version,
        "eval_score":      round(eval_score, 4),
        "prompt_hash":     hash_prompt(system_prompt),
        "trained_at":      datetime.now(timezone.utc).isoformat(),
        "note":            note,
        "active":          True,
    }
    versions = load_versions()
    # Mark all previous entries inactive
    for v in versions:
        v["active"] = False
    versions.append(entry)
    save_versions(versions)
    return entry


def check_prompt_lock(model_id: str, current_prompt: str) -> bool:
    """
    Return True if current_prompt matches the hash stored for model_id.
    Prints a warning if there is a mismatch.

    PROD PATTERN: if prompt changed since training → retrain or revert prompt.
    """
    versions = load_versions()
    for v in versions:
        if v["model_id"] == model_id:
            stored_hash  = v["prompt_hash"]
            current_hash = hash_prompt(current_prompt)
            if stored_hash != current_hash:
                print(f"[VERSION] ⚠  PROMPT–MODEL MISMATCH for {model_id}")
                print(f"[VERSION]    Stored hash  : {stored_hash}")
                print(f"[VERSION]    Current hash : {current_hash}")
                print("[VERSION]    Action required: retrain with current prompt OR revert prompt.")
                return False
            return True
    print(f"[VERSION] Model ID '{model_id}' not found in version registry.")
    return False


def print_versions() -> None:
    versions = load_versions()
    if not versions:
        print("[VERSION] No versions registered yet.")
        return

    print(f"\n{'Model ID':50}  {'Dataset':8}  {'Eval':6}  {'PromptHash':18}  {'Active':6}  Note")
    print("-" * 110)
    for v in versions:
        active = "✓" if v.get("active") else " "
        print(
            f"{v['model_id']:50}  "
            f"{v['dataset_version']:8}  "
            f"{v['eval_score']:6.2%}  "
            f"{v['prompt_hash']:18}  "
            f"{active:6}  "
            f"{v['note']}"
        )


def main():
    from prepare_data import SYSTEM_PROMPT, DATASET_VERSION, hash_prompt as _hp

    fake_model_id = "ft:gpt-4o-mini-2024-07-18:test:ticket-classifier:FAKE001"
    entry = add_version(
        model_id=fake_model_id,
        dataset_version=DATASET_VERSION,
        eval_score=0.90,
        system_prompt=SYSTEM_PROMPT,
        note="Initial fine-tune: support ticket classifier v1.0",
    )
    print(f"[VERSION] Registered model version:")
    print(f"[VERSION]   model_id      : {entry['model_id']}")
    print(f"[VERSION]   dataset       : {entry['dataset_version']}")
    print(f"[VERSION]   eval_score    : {entry['eval_score']:.1%}")
    print(f"[VERSION]   prompt_hash   : {entry['prompt_hash']}")
    print(f"[VERSION]   trained_at    : {entry['trained_at']}")
    print()

    # Demonstrate lock check
    print("[VERSION] Lock check with CORRECT prompt — should pass:")
    check_prompt_lock(fake_model_id, SYSTEM_PROMPT)

    modified_prompt = SYSTEM_PROMPT + " Always respond in uppercase."
    print("\n[VERSION] Lock check with MODIFIED prompt — should warn:")
    check_prompt_lock(fake_model_id, modified_prompt)

    print()
    print_versions()


if __name__ == "__main__":
    main()
