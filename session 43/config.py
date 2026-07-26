"""Config File Separation — every setting lives here, imported everywhere else."""
import os

# Secrets via environment variables — never hardcode the key.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Model + limits (overridable via env, sensible defaults baked in).
MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "8000"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "400"))

# Offline teaching mode — set FAKE_LLM=1 to run with no API key (canned reply).
FAKE = os.environ.get("FAKE_LLM", "0") == "1"
