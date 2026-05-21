"""
Session 22 — loaders.py

Extends the S20 loaders with per-document **metadata**. Today's retrievers —
particularly the metadata-filter retriever — need each chunk to know what kind
of document it came from (manual / paper / blog) and, where possible, which
section. Metadata is collected at INGEST time. This is the production rule:
collect it when you load, never retrofit it later.

Returns: a list of Doc objects, where every Doc has:
    - source:    filename (string)
    - text:      cleaned full-document text
    - metadata:  dict — at minimum `source_type`, plus loader-specific extras

Production patterns reinforced:
  - try/except on every external call (PDF parse, HTML decode)
  - metadata as an ingest-time obligation
  - pipeline logging with [loaders] prefix
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from pypdf import PdfReader
from bs4 import BeautifulSoup

# ── Named constants ─────────────────────────────────────────────────────────
PDF_FOOTER_PATTERNS = [
    r"Page \d+ of \d+.*",
    r"©.*Acme Corp.*",
    r"Document Ref:.*",
]
HTML_TAGS_TO_DROP = ["nav", "footer", "header", "aside",
                     "script", "style", "noscript"]
HTML_CLASSES_TO_DROP = ["site-header", "site-footer", "sidebar",
                        "cookie-banner", "ad-slot", "search-bar",
                        "primary-nav", "footer-nav"]

# Default metadata mapping by filename prefix. In a real system this would
# come from a manifest. For the Phase 4 corpus we know exactly what each file is.
KNOWN_SOURCE_TYPES = {
    "intro_to_rag":   "paper",
    "product_manual": "manual",
    "sample_article": "blog",
}


@dataclass
class Doc:
    """A loaded document plus its metadata. The unit returned by load_corpus."""
    source: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── PDF loader ──────────────────────────────────────────────────────────────

def load_pdf(path: str) -> Doc:
    """Load a PDF, strip page footers, attach metadata."""
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
    name = os.path.basename(path)
    print(f"[loaders] [PDF]  {name:<24}  {len(reader.pages)} pages  ->  "
          f"{len(full):>6} chars")

    return Doc(
        source=name,
        text=full,
        metadata={
            "source_type": _infer_source_type(name),
            "format":      "pdf",
            "page_count":  len(reader.pages),
        },
    )


def _clean_pdf_text(text: str) -> str:
    for pat in PDF_FOOTER_PATTERNS:
        text = re.sub(pat, "", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ── HTML loader ─────────────────────────────────────────────────────────────

def load_html_file(path: str) -> Doc:
    """Load an HTML file, drop chrome, attach metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"HTML not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            html = f.read()

    cleaned = _clean_html(html)
    name = os.path.basename(path)
    print(f"[loaders] [HTML] {name:<24}  {len(html):>6} raw  ->  "
          f"{len(cleaned):>6} clean chars")

    # Pull a publish year from the page if a meta tag is present — illustrative.
    soup = BeautifulSoup(html, "html.parser")
    year_tag = soup.find("meta", attrs={"name": "publish-year"})
    year = int(year_tag["content"]) if year_tag and year_tag.get("content", "").isdigit() else None

    return Doc(
        source=name,
        text=cleaned,
        metadata={
            "source_type": _infer_source_type(name),
            "format":      "html",
            "year":        year,
        },
    )


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in HTML_TAGS_TO_DROP:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for cls in HTML_CLASSES_TO_DROP:
        for tag in soup.find_all(class_=cls):
            tag.decompose()
    main = soup.find("main") or soup.find("article") or soup
    text = main.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Plain text loader ───────────────────────────────────────────────────────

def load_text(path: str) -> Doc:
    """Load a plain-text file. Metadata is sparse — that's the point of .txt."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Text file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    name = os.path.basename(path)
    print(f"[loaders] [TXT]  {name:<24}  {len(text):>6} chars")
    return Doc(
        source=name,
        text=text.strip(),
        metadata={
            "source_type": _infer_source_type(name),
            "format":      "txt",
        },
    )


# ── Corpus dispatcher ───────────────────────────────────────────────────────

def load_corpus(paths: List[str]) -> List[Doc]:
    """Load every file in `paths`, return a list of Doc with metadata attached."""
    docs: List[Doc] = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            docs.append(load_pdf(p))
        elif ext in (".html", ".htm"):
            docs.append(load_html_file(p))
        elif ext == ".txt":
            docs.append(load_text(p))
        else:
            print(f"[loaders] WARN: skipping unknown extension: {p}")
    return docs


def _infer_source_type(filename: str) -> str:
    """Map a filename to a coarse source_type. Production systems use a manifest."""
    stem = os.path.splitext(filename)[0]
    for key, label in KNOWN_SOURCE_TYPES.items():
        if stem.startswith(key):
            return label
    return "unknown"


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    files = [
        os.path.join(here, "data", "intro_to_rag.pdf"),
        os.path.join(here, "data", "product_manual.pdf"),
        os.path.join(here, "data", "sample_article.html"),
    ]
    docs = load_corpus(files)
    for d in docs:
        print("─" * 70)
        print(f"[{d.source}]  metadata={d.metadata}")
        print(d.text[:300] + ("..." if len(d.text) > 300 else ""))
