"""
Session 41 — Fine-Tuning Fundamentals: When and How
Module 4 of 7: monitor_job.py

Polls fine-tuning job status and prints loss curves.
Detects overfitting: training_loss falling while validation_loss rising.

FAKE_LLM=1  OR  job_id starts with "ftjob-FAKE": simulates a 3-epoch run
where overfitting fires in epoch 3.
"""

import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, field

CONFIG_PATH  = Path(os.environ.get("FT_CONFIG_PATH", "ft_config.json"))
POLL_SECONDS = int(os.environ.get("FT_POLL_SECONDS", "60"))
OVERFIT_GAP  = float(os.environ.get("FT_OVERFIT_GAP", "0.05"))  # val > train + gap


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    trained_tokens: int = 0


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def save_config(data: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


def _print_curves(metrics: list[EpochMetrics]) -> None:
    print(f"\n{'Epoch':>6}  {'train_loss':>12}  {'val_loss':>12}  {'tokens':>10}  {'status':>18}")
    print("-" * 68)
    for m in metrics:
        overfit = m.val_loss > m.train_loss + OVERFIT_GAP
        status  = "⚠ OVERFIT SIGNAL" if overfit else "OK"
        print(f"{m.epoch:>6}  {m.train_loss:>12.4f}  {m.val_loss:>12.4f}  {m.trained_tokens:>10,}  {status:>18}")


def _detect_overfitting(metrics: list[EpochMetrics]) -> bool:
    if len(metrics) < 2:
        return False
    last = metrics[-1]
    return last.val_loss > last.train_loss + OVERFIT_GAP


def _simulate_metrics() -> list[EpochMetrics]:
    """
    Simulated loss curves for fake mode.
    Epoch 1 and 2 look healthy. Epoch 3 shows overfitting.
    """
    return [
        EpochMetrics(epoch=1, train_loss=1.2840, val_loss=1.2910, trained_tokens=25_000),
        EpochMetrics(epoch=2, train_loss=0.8310, val_loss=0.8490, trained_tokens=25_000),
        EpochMetrics(epoch=3, train_loss=0.3420, val_loss=0.5180, trained_tokens=25_000),
    ]


def _is_fake(job_id: str) -> bool:
    """Fake mode if FAKE_LLM=1, no API key, or job_id is a placeholder."""
    return (
        os.environ.get("FAKE_LLM", "0") == "1"
        or not os.environ.get("OPENAI_API_KEY", "")
        or job_id.startswith("ftjob-FAKE")
    )


def poll_job(job_id: str) -> tuple[str, list[EpochMetrics]]:
    """
    Poll until job is succeeded/failed.
    Returns (final_model_id_or_empty, metrics_list).
    """
    if _is_fake(job_id):
        print(f"[MONITOR] Offline mode — simulating 3-epoch training run")
        print(f"[MONITOR] Job ID : {job_id}")
        metrics = _simulate_metrics()
        _print_curves(metrics)

        overfit = _detect_overfitting(metrics)
        if overfit:
            print("\n[MONITOR] ⚠  OVERFITTING DETECTED in epoch 3")
            print("[MONITOR]    val_loss rose above train_loss + threshold.")
            print("[MONITOR]    Consider: reduce n_epochs=1 or add more diverse data.")
        else:
            print("\n[MONITOR] ✓ Loss curves healthy — no overfitting detected.")

        fake_model_id = "ft:gpt-4o-mini-2024-07-18:test:ticket-classifier:FAKE001"
        print(f"\n[MONITOR] Status          : succeeded (simulated)")
        print(f"[MONITOR] fine_tuned_model : {fake_model_id}")
        return fake_model_id, metrics

    # Real polling path
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        metrics: list[EpochMetrics] = []

        while True:
            job = client.fine_tuning.jobs.retrieve(job_id)
            print(f"[MONITOR] Status: {job.status}")

            # Fetch latest events for loss data
            events = client.fine_tuning.jobs.list_events(
                fine_tuning_job_id=job_id, limit=20
            )
            for ev in reversed(events.data):
                if ev.type == "metrics" and hasattr(ev, "data"):
                    d = ev.data
                    m = EpochMetrics(
                        epoch=int(d.get("step", 0)),
                        train_loss=float(d.get("train_loss", 0)),
                        val_loss=float(d.get("valid_loss", 0)),
                        trained_tokens=int(d.get("total_tokens", 0)),
                    )
                    if not any(x.epoch == m.epoch for x in metrics):
                        metrics.append(m)

            if metrics:
                _print_curves(sorted(metrics, key=lambda x: x.epoch))
                if _detect_overfitting(metrics):
                    print("\n[MONITOR] ⚠  OVERFITTING DETECTED — val_loss diverging from train_loss")

            if job.status in ("succeeded", "failed", "cancelled"):
                model_id = getattr(job, "fine_tuned_model", "") or ""
                print(f"\n[MONITOR] Final model : {model_id or '(none — job did not succeed)'}")
                return model_id, metrics

            print(f"[MONITOR] Waiting {POLL_SECONDS}s before next poll...")
            time.sleep(POLL_SECONDS)

    except ImportError:
        raise ImportError("openai package not installed.")
    except KeyError:
        raise EnvironmentError("OPENAI_API_KEY not set.")


def main() -> str:
    cfg = load_config()
    job_id = cfg.get("job_id", "")
    if not job_id:
        print("[MONITOR] No job_id found. Run 'python demo.py start' first.")
        return ""

    model_id, metrics = poll_job(job_id)

    if model_id:
        cfg["fine_tuned_model"] = model_id
        cfg["metrics"] = [
            {"epoch": m.epoch, "train_loss": m.train_loss, "val_loss": m.val_loss}
            for m in metrics
        ]
        save_config(cfg)
        print(f"[MONITOR] Saved fine_tuned_model to {CONFIG_PATH}")

    return model_id


if __name__ == "__main__":
    main()
