# Session 41 — Fine-Tuning Fundamentals: When and How
## Reference Code

7 modules, 4 production patterns, full offline support via `FAKE_LLM=1`.

### Quick start (offline, no API key)
```bash
cd Session_41/Code
pip install -r requirements.txt
FAKE_LLM=1 PYTHONPYCACHEPREFIX=/tmp/s41_pycache python demo.py all
```

### Demo modes
| Command | What it does |
|---|---|
| `python demo.py prepare` | Build + validate training JSONL (5 quality rules) |
| `python demo.py upload`  | Upload file to OpenAI Files API → save file_id |
| `python demo.py monitor` | Poll job, print loss curves, detect overfitting |
| `python demo.py eval`    | Compare base vs fine-tuned model, apply eval gate |
| `python demo.py versions`| Show version registry + prompt–model lock demo |
| `python demo.py all`     | Run all five in sequence |

### 4 Production Patterns
1. **Fine-Tuned Model Versioning** — `version_config.py`: tag every model ID with date, dataset, eval score
2. **Eval Gate Before Deploy** — `run_eval.py`: GATE PASS required before updating FT_MODEL_ID
3. **Training Data Provenance** — `prepare_data.py` + `version_config.py`: JSONL files version-controlled, each model entry records dataset version
4. **Prompt–Model Version Lock** — `version_config.py`: system prompt hash stored alongside model ID; mismatch triggers retrain warning
