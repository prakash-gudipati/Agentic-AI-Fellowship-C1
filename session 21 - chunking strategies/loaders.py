
import os
import re
from typing import List, Tuple

from pypdf import PdfReader
from bs4 import BeautifulSoup

# ── Named constants ─────────────────────────────────────────────────────────
PDF_FOOTER_PATTERNS = [
    r"Page \d+ of \d+.*",
    r"©.*Acme Corp.*",
    r"Document Ref:.*",
]
HTML_TAGS_TO_DROP = ["nav", "footer", "header", "aside", "script", "style", "noscript"]
HTML_CLASSES_TO_DROP = ["site-header", "site-footer", "sidebar", "cookie-banner",
                        "ad-slot", "search-bar", "primary-nav", "footer-nav"]


# ── PDF loader ──────────────────────────────────────────────────────────────

def load_pdf(path: str) -> str:
    """Load a PDF, strip page-level footers, return clean text.

    Uses pypdf to extract per-page text, then runs simple regex cleanup
    on each page before concatenation.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        reader = PdfReader(path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF {path}: {e}") from e

    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception as e:
            print(f"[loaders] WARN: page {i+1} of {path} failed to extract: {e}")
            continue
        pages.append(_clean_pdf_text(raw))

    full = "\n\n".join(p for p in pages if p.strip())
    print(f"[loaders] [PDF]  {os.path.basename(path):<24}  "
          f"{len(reader.pages)} pages  →  {len(full):>6} chars")
    return full


def _clean_pdf_text(text: str) -> str:
    """Strip footer patterns and collapse whitespace."""
    for pat in PDF_FOOTER_PATTERNS:
        text = re.sub(pat, "", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ── HTML loader ─────────────────────────────────────────────────────────────

def load_html_file(path: str) -> str:
    """Load an HTML file from disk and return clean article text."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"HTML not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
    except UnicodeDecodeError:
        # Fallback for badly-encoded pages — production reality.
        with open(path, "r", encoding="latin-1") as f:
            html = f.read()
    cleaned = _clean_html(html)
    print(f"[loaders] [HTML] {os.path.basename(path):<24}  "
          f"{len(html):>6} raw  →  {len(cleaned):>6} clean chars")
    return cleaned


def _clean_html(html: str) -> str:
    """Drop nav / footer / sidebar / ads, keep the article."""
    soup = BeautifulSoup(html, "html.parser")

    # Strip whole tags by name.
    for tag_name in HTML_TAGS_TO_DROP:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Strip elements by class name.
    for cls in HTML_CLASSES_TO_DROP:
        for tag in soup.find_all(class_=cls):
            tag.decompose()

    # Prefer <main> or <article> if present — they hold the real content.
    main = soup.find("main") or soup.find("article") or soup
    text = main.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Plain text loader ───────────────────────────────────────────────────────

def load_text(path: str) -> str:
    """Load a plain-text file. Used for completeness — most corpora include .txt."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Text file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"[loaders] [TXT]  {os.path.basename(path):<24}  "
          f"{len(text):>6} chars")
    return text.strip()


# ── Corpus dispatcher ───────────────────────────────────────────────────────

def load_corpus(paths: List[str]) -> List[Tuple[str, str]]:
    """Load every file in `paths`, dispatching by extension.
    Returns a list of (source_filename, clean_text) tuples.
    """
    docs: List[Tuple[str, str]] = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            text = load_pdf(p)
        elif ext in (".html", ".htm"):
            text = load_html_file(p)
        elif ext == ".txt":
            text = load_text(p)
        else:
            print(f"[loaders] WARN: skipping unknown extension: {p}")
            continue
        if text:
            docs.append((os.path.basename(p), text))
    return docs


if __name__ == "__main__":
    # Quick smoke test — run python loaders.py
    here = os.path.dirname(__file__)
    files = [
        os.path.join(here, "data", "intro_to_rag.pdf"),
        os.path.join(here, "data", "product_manual.pdf"),
        os.path.join(here, "data", "sample_article.html"),
    ]
    docs = load_corpus(files)
    for name, text in docs:
        print("─" * 70)
        print(f"[{name}]  preview:")
        print(text[:400] + ("..." if len(text) > 400 else ""))
