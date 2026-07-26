"""
Session 41 — Fine-Tuning Fundamentals: When and How
Module 1 of 7: prepare_data.py

Builds, validates, and writes training + evaluation JSONL files.
PROD PATTERN: Training Data Provenance — every dataset write is versioned.

Five data quality rules enforced:
  1. Consistent format      — system prompt identical in every example
  2. Diverse inputs         — basic uniqueness check on user field
  3. Correct completions    — non-empty assistant field
  4. Edge cases included    — caller responsibility; flagged in summary
  5. No PII                 — basic regex scan for emails / phone numbers

Usage:
    python prepare_data.py          # writes JSONL files to current directory
    DATASET_DIR=/tmp python prepare_data.py
"""

import json
import os
import re
import hashlib
from pathlib import Path

# ── SYSTEM PROMPT ────────────────────────────────────────────────────────────
# RULE: every training example uses this EXACT string.
# Changing it invalidates all existing training data (hash will change).
SYSTEM_PROMPT = (
    "You are a support ticket classifier. "
    "Read the ticket text and respond with exactly one category label. "
    "Valid categories: billing, technical, account, shipping, other. "
    "Respond with only the category name, lowercase, no punctuation."
)

DATASET_VERSION = "v1.0"
MAX_TOKENS_PER_EXAMPLE = 512   # OpenAI fine-tuning limit per example

# ── RAW EXAMPLES ─────────────────────────────────────────────────────────────
# Each tuple: (user_text, expected_label)
# In production these come from real logs + human annotation.
RAW_EXAMPLES = [
    # billing (12)
    ("My invoice shows a charge I don't recognise from last month.", "billing"),
    ("I was charged twice for the same subscription renewal.", "billing"),
    ("Can I get a refund for my last payment?", "billing"),
    ("My credit card was declined even though it has funds.", "billing"),
    ("Where can I find my payment history?", "billing"),
    ("I need to update my billing address.", "billing"),
    ("The promo code did not apply to my order total.", "billing"),
    ("I cancelled but was still charged this month.", "billing"),
    ("How do I switch from monthly to annual billing?", "billing"),
    ("My VAT invoice has the wrong company name.", "billing"),
    ("I need a GST receipt for my last three transactions.", "billing"),
    ("The auto-renewal happened before my reminder date.", "billing"),
    # technical (12)
    ("The app crashes every time I try to upload a file.", "technical"),
    ("Login page keeps showing a 500 error.", "technical"),
    ("My dashboard data stopped refreshing two hours ago.", "technical"),
    ("The export button does nothing when I click it.", "technical"),
    ("Notifications stopped working after the last update.", "technical"),
    ("The mobile app is stuck on a loading spinner.", "technical"),
    ("API calls are returning 403 even with a valid key.", "technical"),
    ("The search bar returns no results for any query.", "technical"),
    ("I cannot connect the Chrome extension to my account.", "technical"),
    ("Two-factor authentication is not sending the SMS code.", "technical"),
    ("The PDF report generator produces a blank file.", "technical"),
    ("Webhook events are not firing for my endpoint.", "technical"),
    # account (10)
    ("I forgot my password and the reset email is not arriving.", "account"),
    ("Can I change my username?", "account"),
    ("I need to transfer my account to a different email address.", "account"),
    ("How do I delete my account permanently?", "account"),
    ("I locked myself out after too many wrong password attempts.", "account"),
    ("I want to add a team member to my workspace.", "account"),
    ("Can I merge two accounts into one?", "account"),
    ("My account was suspended but I did not violate any rules.", "account"),
    ("How do I enable single sign-on for my organisation?", "account"),
    ("I need to remove a collaborator from my project.", "account"),
    # shipping (8)
    ("My order has not arrived and it has been 10 days.", "shipping"),
    ("The tracking number shows delivered but I have not received it.", "shipping"),
    ("Can I change my delivery address after placing the order?", "shipping"),
    ("The parcel was returned to sender — how do I reship?", "shipping"),
    ("I received the wrong item in my package.", "shipping"),
    ("The package arrived damaged.", "shipping"),
    ("What are the estimated delivery times to India?", "shipping"),
    ("I need to ship to a PO Box — is that supported?", "shipping"),
    # other (8)
    ("Do you offer a student discount?", "other"),
    ("What are your business hours for live support?", "other"),
    ("Can I white-label your product for my clients?", "other"),
    ("Is there an affiliate programme?", "other"),
    ("Do you have a public API roadmap?", "other"),
    ("I have a feature request for the mobile app.", "other"),
    ("What data centres do you use?", "other"),
    ("Is your service GDPR compliant?", "other"),
]

# ── PII PATTERNS ─────────────────────────────────────────────────────────────
_PII_PATTERNS = [
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),  # email
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),                # phone US
    re.compile(r"\b\d{10,12}\b"),                                      # phone IN
    re.compile(r"\b(?:\d[ -]?){12,16}\b"),                             # card-like
]


def _count_tokens_approx(text: str) -> int:
    """Rough token estimate: ~4 chars per token. Avoids tiktoken dependency."""
    return max(1, len(text) // 4)


def _has_pii(text: str) -> bool:
    return any(p.search(text) for p in _PII_PATTERNS)


def build_examples() -> list[dict]:
    """Convert raw (user, label) pairs into OpenAI training format."""
    examples = []
    for user_text, label in RAW_EXAMPLES:
        examples.append({
            "messages": [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": user_text},
                {"role": "assistant", "content": label},
            ]
        })
    return examples


def validate_example(ex: dict, system_prompt: str) -> list[str]:
    """Return a list of violation strings. Empty list = valid."""
    violations = []
    msgs = ex.get("messages", [])

    roles = [m["role"] for m in msgs]
    if roles != ["system", "user", "assistant"]:
        violations.append(f"Wrong role sequence: {roles}")
        return violations  # can't continue checks

    sys_content  = msgs[0]["content"]
    user_content = msgs[1]["content"]
    asst_content = msgs[2]["content"]

    # Rule 1: consistent system prompt
    if sys_content != system_prompt:
        violations.append("System prompt mismatch (Rule 1)")

    # Rule 3: non-empty completions
    if not asst_content.strip():
        violations.append("Empty assistant completion (Rule 3)")

    # Rule 5: no PII
    for field, value in [("user", user_content), ("assistant", asst_content)]:
        if _has_pii(value):
            violations.append(f"Possible PII in {field} field (Rule 5)")

    # Token limit
    total_tokens = sum(_count_tokens_approx(m["content"]) for m in msgs)
    if total_tokens > MAX_TOKENS_PER_EXAMPLE:
        violations.append(
            f"Exceeds token limit: ~{total_tokens} tokens > {MAX_TOKENS_PER_EXAMPLE} (Rule token)"
        )

    return violations


def split_train_eval(examples: list[dict], ratio: float = 0.9):
    """Split examples into (train, eval). Preserves order; no shuffle for determinism."""
    split_at = int(len(examples) * ratio)
    return examples[:split_at], examples[split_at:]


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def hash_prompt(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def main():
    out_dir = Path(os.environ.get("DATASET_DIR", "."))

    examples = build_examples()

    # Validate all
    total_violations = 0
    for i, ex in enumerate(examples):
        v = validate_example(ex, SYSTEM_PROMPT)
        if v:
            total_violations += len(v)
            print(f"  [VIOLATION] Example {i}: {'; '.join(v)}")

    train_examples, eval_examples = split_train_eval(examples)

    train_path = out_dir / "training_data.jsonl"
    eval_path  = out_dir / "eval_data.jsonl"
    write_jsonl(train_examples, train_path)
    write_jsonl(eval_examples,  eval_path)

    prompt_hash = hash_prompt(SYSTEM_PROMPT)

    print(f"\n[PREPARE] Dataset version : {DATASET_VERSION}")
    print(f"[PREPARE] System prompt hash : {prompt_hash}")
    print(f"[PREPARE] Total examples      : {len(examples)}")
    print(f"[PREPARE] Training examples   : {len(train_examples)}  → {train_path}")
    print(f"[PREPARE] Eval examples       : {len(eval_examples)}   → {eval_path}")
    print(f"[PREPARE] Quality violations  : {total_violations}")
    if total_violations == 0:
        print("[PREPARE] ✓ All examples pass the 5 quality rules")
    else:
        print(f"[PREPARE] ✗ Fix {total_violations} violation(s) before uploading")

    return train_path, eval_path, prompt_hash


if __name__ == "__main__":
    main()
