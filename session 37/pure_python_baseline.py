"""
Session 37 — pure_python_baseline.py

The 'before' picture. This is how the student already knows how to call a
model — the pure-Python way from S29/S30: build the messages dict by hand,
call the SDK, pull the text out of the response, do the string formatting
yourself.

Demo 1 runs this side by side with the LangChain version so the class can
see exactly what LangChain does for free — and what it hides.

This file imports the Anthropic SDK directly (no LangChain) to make the
contrast honest. In FAKE_LLM mode it skips the SDK and returns a canned
reply so the demo runs offline.
"""

from __future__ import annotations

import os

# The canned answer reused from the fake model so the two Demo-1 outputs
# match and the comparison is purely about CODE SHAPE, not content.
from model_factory import _CANNED_SUMMARY


SUMMARISER_INSTRUCTION = (
    "You are a summariser. Condense the user's text into a short, faithful "
    "summary. Keep only facts that appear in the text."
)


def summarise_pure_python(document: str) -> str:
    """Summarise text the manual way — no framework.

    Every step is explicit: assemble the system+user messages, call the
    API, then reach into the response object to extract the text. This is
    the boilerplate LangChain collapses into `prompt | model | parser`.
    """

    if os.environ.get("FAKE_LLM", "") == "1":
        # Offline path — pretend the call happened.
        return _CANNED_SUMMARY

    try:
        import anthropic  # type: ignore
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "anthropic SDK not installed. 'pip install anthropic' or set "
            "FAKE_LLM=1."
        ) from exc

    client = anthropic.Anthropic()

    # 1) You build the message structure by hand.
    user_text = f"Summarise the following text:\n\n{document}"

    # 2) You call the API and pass every parameter yourself.
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        temperature=0,
        system=SUMMARISER_INSTRUCTION,
        messages=[{"role": "user", "content": user_text}],
    )

    # 3) You dig the text out of the response object yourself.
    parts = [block.text for block in response.content if block.type == "text"]
    return "".join(parts).strip()


# A rough line count of the *essential* steps above — used by Demo 1 to
# print the 'lines of glue code' contrast against the LCEL one-liner.
ESSENTIAL_STEP_COUNT = 3
