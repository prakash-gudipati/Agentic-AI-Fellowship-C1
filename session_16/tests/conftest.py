"""pytest fixtures for the S16 LLMClient test suite."""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=True)
def require_keys():
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("Set at least one API key in .env to run these tests")
