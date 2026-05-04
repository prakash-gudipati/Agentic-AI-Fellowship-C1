"""
pytest fixtures for the LLMClient test suite.

Shared setup: load .env, build a real client, and provide convenient
patching helpers so individual tests stay short.
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=True)
def require_keys():
    """Skip tests that need keys when keys are not configured.

    autouse=True means this runs before every test, so contributors
    without keys see a clean SKIP rather than a confusing failure.
    """
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("Set at least one API key in .env to run these tests")
