import html
import re
import ftfy

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
SMART_QUOTE_MAP = {
    "‘": "'",   # left single quote
    "’": "'",   # right single quote
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "–": "-",   # en dash
    "—": "-",   # em dash
    "…": "...", # horizontal ellipsis
}
PARAGRAPH_SEPERATOR = re.compile(r"\n\s*\n")
SENTENCE_SPLIT = re.compile(r"(?=[.!?])\s+(?=[A-Z])")

def clean_whitespace(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "|n\n", text)
    return text

def fix_encoding(text: str) -> str:
    return ftfy.fix_text(text)

def strip_html(text: str) -> str:
    text = HTML_TAG_PATTERN.sub("",text)
    text = html.unescape(text)
    return text

def normalize_quotes(text:str) -> str:
    for smart, plain in SMART_QUOTE_MAP.items():
        text = text.replace(smart,plain)
        return text
    
def split_paragraphs(text:str) -> str:
    parts =  PARAGRAPH_SEPERATOR.split(text)
    return [p.strip() for p in parts if p.strip()]

def split_sentences(text:str) -> str:
    parts =  SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]

def clean_text(text: str) -> str:
    text = strip_html(text)
    text = fix_encoding(text)
    text = normalize_quotes(text)
    text = clean_whitespace(text)
    return text