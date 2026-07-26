"""summarizer.py — the one function that calls Claude and returns a clean dict."""
import json
import config

SYSTEM = (
    "You are a text summarizer. Read the user's text and reply with ONLY a JSON "
    "object, no prose, in exactly this shape:\n"
    '{"tl_dr": "<one or two sentence summary>", '
    '"key_points": ["point one", "point two", "point three"]}\n'
    "key_points must contain exactly 3 short strings."
)


def _fallback(text: str) -> dict:
    """If the model returns something unparseable, never crash the endpoint."""
    snippet = " ".join(text.split())[:160]
    return {
        "tl_dr": snippet or "No text provided.",
        "key_points": [
            "Could not parse a structured summary.",
            "Showing a trimmed snippet of the input instead.",
            "Try again with cleaner text.",
        ],
    }


def _fake(text: str) -> dict:
    """Offline canned reply for FAKE_LLM=1 (no API key needed)."""
    words = len(text.split())
    return {
        "tl_dr": f"This is a demo summary of a {words}-word passage about the topic you pasted.",
        "key_points": [
            "First key idea pulled from the text.",
            "Second key idea pulled from the text.",
            "Third key idea pulled from the text.",
        ],
    }


def _coerce(data: dict, text: str) -> dict:
    """Guarantee the contract: tl_dr str + exactly 3 key_points."""
    tl_dr = str(data.get("tl_dr", "")).strip()
    pts = data.get("key_points") or []
    pts = [str(p).strip() for p in pts if str(p).strip()]
    pts = (pts + ["", "", ""])[:3]
    if not tl_dr:
        return _fallback(text)
    return {"tl_dr": tl_dr, "key_points": pts}


def summarize(text: str) -> dict:
    text = (text or "").strip()[: config.MAX_INPUT_CHARS]
    if not text:
        return {"tl_dr": "No text provided.", "key_points": ["", "", ""]}

    if config.FAKE:
        return _fake(text)

    from anthropic import Anthropic  # imported here so FAKE mode needs no install

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=config.MODEL,
        max_tokens=config.MAX_OUTPUT_TOKENS,
        system=SYSTEM,
        messages=[
            {"role": "user", "content": f"Summarize this text:\n\n{text}"},
            {"role": "assistant", "content": "{"},  # prefill: force JSON
        ],
    )
    raw = "{" + msg.content[0].text
    try:
        return _coerce(json.loads(raw), text)
    except Exception:
        return _fallback(text)
