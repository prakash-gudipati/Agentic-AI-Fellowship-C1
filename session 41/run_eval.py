"""
Session 41 — Fine-Tuning Fundamentals: When and How
Module 5 of 7: run_eval.py

PROD PATTERN: Eval Gate Before Deploy.
Compares base model vs fine-tuned model on held-out eval data.
Prints side-by-side comparison and applies a pass/fail gate.

Flags:
  --ci-mode   exit 0 on GATE PASS, exit 1 on GATE FAIL (for CI pipelines)

FAKE_LLM=1: uses deterministic canned responses — fine-tuned model always wins.
"""

import json
import os
import sys
from pathlib import Path

CONFIG_PATH  = Path(os.environ.get("FT_CONFIG_PATH", "ft_config.json"))
EVAL_PATH    = Path(os.environ.get("EVAL_DATA_PATH", "eval_data.jsonl"))
GATE_THRESHOLD = float(os.environ.get("EVAL_GATE_THRESHOLD", "0.75"))


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def load_eval_examples(path: Path) -> list[dict]:
    if not path.exists():
        return []
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _fake_base_response(user_text: str) -> str:
    """Simulates a base model that makes 25% errors."""
    errors = {
        "I was charged twice for the same subscription renewal.": "technical",
        "My order has not arrived and it has been 10 days.": "account",
    }
    # Rough heuristic — reasonable but imperfect
    user_lower = user_text.lower()
    if user_text in errors:
        return errors[user_text]  # intentional wrong answer
    if any(w in user_lower for w in ["bill", "charge", "payment", "invoice", "refund", "gst", "vat"]):
        return "billing"
    if any(w in user_lower for w in ["crash", "error", "api", "export", "loading", "webhook", "pdf"]):
        return "technical"
    if any(w in user_lower for w in ["password", "account", "login", "email", "sso", "team"]):
        return "account"
    if any(w in user_lower for w in ["order", "deliver", "ship", "parcel", "package"]):
        return "shipping"
    return "other"


def _fake_ft_response(user_text: str) -> str:
    """Simulates a fine-tuned model that makes ~5% errors (near-perfect)."""
    user_lower = user_text.lower()
    if any(w in user_lower for w in ["bill", "charge", "payment", "invoice", "refund", "gst", "vat", "promo", "receipt", "annual", "monthly", "cancel"]):
        return "billing"
    if any(w in user_lower for w in ["crash", "error", "api", "export", "loading", "webhook", "pdf", "search", "extension", "sms", "2fa", "500", "403", "button", "dashboard", "notification"]):
        return "technical"
    if any(w in user_lower for w in ["password", "account", "login", "email", "sso", "team", "merge", "suspend", "collaborator", "username", "delete"]):
        return "account"
    if any(w in user_lower for w in ["order", "deliver", "ship", "parcel", "package", "tracking", "damaged", "wrong item", "reship", "po box"]):
        return "shipping"
    return "other"


def call_model(messages: list[dict], model_id: str, fake: bool) -> str:
    """Call a model and return its completion text."""
    if fake:
        user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
        base_model_prefix = "gpt-4o-mini"
        if model_id.startswith("ft:"):
            return _fake_ft_response(user_text)
        else:
            return _fake_base_response(user_text)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=10,
            temperature=0,
        )
        return resp.choices[0].message.content.strip().lower()
    except ImportError:
        raise ImportError("openai package not installed.")


def run_eval(base_model: str, ft_model: str, examples: list[dict], fake: bool) -> dict:
    """
    Run both models on all examples. Return metrics dict.
    """
    base_correct = 0
    ft_correct   = 0
    results = []

    print(f"\n{'Input (truncated)':40}  {'Expected':10}  {'Base':10}  {'FT':10}  {'Δ'}")
    print("-" * 78)

    for ex in examples:
        msgs     = ex["messages"]
        expected = msgs[2]["content"].strip().lower()
        user_txt = msgs[1]["content"]

        base_pred = call_model(msgs[:2], base_model, fake).strip().lower()
        ft_pred   = call_model(msgs[:2], ft_model,   fake).strip().lower()

        base_ok = base_pred == expected
        ft_ok   = ft_pred   == expected
        if base_ok:
            base_correct += 1
        if ft_ok:
            ft_correct += 1

        delta = ""
        if ft_ok and not base_ok:
            delta = "✓ FT wins"
        elif base_ok and not ft_ok:
            delta = "✗ FT regressed"

        print(f"{user_txt[:40]:40}  {expected:10}  {base_pred:10}  {ft_pred:10}  {delta}")

    n = len(examples)
    base_acc = base_correct / n if n else 0
    ft_acc   = ft_correct   / n if n else 0

    return {
        "n":         n,
        "base_acc":  base_acc,
        "ft_acc":    ft_acc,
        "base_correct": base_correct,
        "ft_correct":   ft_correct,
    }


def main(ci_mode: bool = False) -> bool:
    cfg  = load_config()

    base_model = cfg.get("base_model", "gpt-4o-mini-2024-07-18")
    ft_model   = cfg.get("fine_tuned_model", "")
    if not ft_model:
        # Use the env var as fallback (PROD PATTERN: Version Lock)
        ft_model = os.environ.get("FT_MODEL_ID", "")

    fake = (
        os.environ.get("FAKE_LLM", "0") == "1"
        or not os.environ.get("OPENAI_API_KEY", "")
        or "FAKE" in str(ft_model)
        or "FAKE" in str(base_model)
    )
    if not ft_model:
        if fake:
            ft_model = "ft:gpt-4o-mini-2024-07-18:test:ticket-classifier:FAKE001"
        else:
            print("[EVAL] No fine_tuned_model found. Run 'python demo.py monitor' first.")
            if ci_mode:
                sys.exit(1)
            return False

    examples = load_eval_examples(EVAL_PATH)
    if not examples:
        print(f"[EVAL] No eval examples found at {EVAL_PATH}. Run 'python demo.py prepare' first.")
        if ci_mode:
            sys.exit(1)
        return False

    print(f"[EVAL] Base model      : {base_model}")
    print(f"[EVAL] Fine-tuned model: {ft_model}")
    print(f"[EVAL] Eval examples   : {len(examples)}")

    metrics = run_eval(base_model, ft_model, examples, fake)

    print(f"\n{'─' * 50}")
    print(f"[EVAL] Base model accuracy  : {metrics['base_acc']:.1%}  ({metrics['base_correct']}/{metrics['n']})")
    print(f"[EVAL] Fine-tuned accuracy  : {metrics['ft_acc']:.1%}  ({metrics['ft_correct']}/{metrics['n']})")
    print(f"[EVAL] Improvement          : {metrics['ft_acc'] - metrics['base_acc']:+.1%}")
    print(f"{'─' * 50}")

    gate_pass = metrics["ft_acc"] >= GATE_THRESHOLD
    if gate_pass:
        print(f"\n[EVAL] ✓ GATE PASS  (ft_acc {metrics['ft_acc']:.1%} ≥ threshold {GATE_THRESHOLD:.0%})")
        print("[EVAL]   → Safe to deploy. Update FT_MODEL_ID in your config.")
    else:
        print(f"\n[EVAL] ✗ GATE FAIL  (ft_acc {metrics['ft_acc']:.1%} < threshold {GATE_THRESHOLD:.0%})")
        print("[EVAL]   → Do NOT deploy. Diagnose training data and retrain.")

    if ci_mode:
        sys.exit(0 if gate_pass else 1)

    return gate_pass


if __name__ == "__main__":
    ci = "--ci-mode" in sys.argv
    main(ci_mode=ci)
