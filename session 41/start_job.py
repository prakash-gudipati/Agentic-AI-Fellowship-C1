"""
Session 41 — Fine-Tuning Fundamentals: When and How
Module 3 of 7: start_job.py

Launches an OpenAI fine-tuning job from an uploaded file_id.

FAKE_LLM=1  OR  file_id starts with "file-FAKE": skips real API call.
"""

import json
import os
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("FT_CONFIG_PATH", "ft_config.json"))
BASE_MODEL  = os.environ.get("FT_BASE_MODEL", "gpt-4o-mini-2024-07-18")
N_EPOCHS    = int(os.environ.get("FT_N_EPOCHS", "3"))


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def save_config(data: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


def _is_fake(file_id: str) -> bool:
    return (
        os.environ.get("FAKE_LLM", "0") == "1"
        or not os.environ.get("OPENAI_API_KEY", "")
        or file_id.startswith("file-FAKE")
    )


def start_job(file_id: str) -> str:
    if _is_fake(file_id):
        job_id = "ftjob-FAKE001xyz789"
        print(f"[START] Offline mode — skipping real API call")
        print(f"[START] file_id    : {file_id}")
        print(f"[START] base_model : {BASE_MODEL}")
        print(f"[START] n_epochs   : {N_EPOCHS}")
        print(f"[START] job_id     : {job_id}")
        cfg = load_config()
        cfg["job_id"] = job_id
        cfg["base_model"] = BASE_MODEL
        save_config(cfg)
        print(f"[START] Saved job_id to {CONFIG_PATH}")
        return job_id

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        job = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=BASE_MODEL,
            hyperparameters={"n_epochs": N_EPOCHS},
        )
        job_id = job.id
        print(f"[START] job_id     : {job_id}")
        cfg = load_config()
        cfg["job_id"] = job_id
        cfg["base_model"] = BASE_MODEL
        save_config(cfg)
        print(f"[START] Saved job_id to {CONFIG_PATH}")
        return job_id
    except ImportError:
        raise ImportError("openai package not installed.")
    except KeyError:
        raise EnvironmentError("OPENAI_API_KEY not set.")


def main() -> str:
    cfg = load_config()
    file_id = cfg.get("file_id", "")
    if not file_id:
        print("[START] No file_id found. Run 'python demo.py upload' first.")
        return ""
    return start_job(file_id)


if __name__ == "__main__":
    main()
