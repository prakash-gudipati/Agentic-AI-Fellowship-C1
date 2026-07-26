"""
Session 41 — Fine-Tuning Fundamentals: When and How
Module 2 of 7: upload_file.py

Uploads the training .jsonl file to the OpenAI Files API and saves the file_id.

FAKE_LLM=1  OR  train_path contains "fake"/"FAKE": skips real API call.
"""

import json
import os
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("FT_CONFIG_PATH", "ft_config.json"))


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def save_config(data: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


def _is_fake(train_path: str = "") -> bool:
    return (
        os.environ.get("FAKE_LLM", "0") == "1"
        or not os.environ.get("OPENAI_API_KEY", "")
        or "fake" in str(train_path).lower()
    )


def upload_file(train_path: str) -> str:
    if _is_fake(train_path):
        file_id = "file-FAKE001abc123"
        print(f"[UPLOAD] Offline mode — skipping real API call")
        print(f"[UPLOAD] Train file : {train_path}")
        print(f"[UPLOAD] file_id    : {file_id}")
        cfg = load_config()
        cfg["file_id"] = file_id
        cfg["train_path"] = str(train_path)
        save_config(cfg)
        print(f"[UPLOAD] Saved file_id to {CONFIG_PATH}")
        return file_id

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        with open(train_path, "rb") as f:
            response = client.files.create(file=f, purpose="fine-tune")
        file_id = response.id
        print(f"[UPLOAD] file_id    : {file_id}")
        cfg = load_config()
        cfg["file_id"] = file_id
        cfg["train_path"] = str(train_path)
        save_config(cfg)
        print(f"[UPLOAD] Saved file_id to {CONFIG_PATH}")
        return file_id
    except ImportError:
        raise ImportError("openai package not installed.")
    except KeyError:
        raise EnvironmentError("OPENAI_API_KEY not set.")


def main(train_path: str = "") -> str:
    cfg = load_config()
    if not train_path:
        train_path = cfg.get("train_path", "training_data.jsonl")
    return upload_file(train_path)


if __name__ == "__main__":
    main()
