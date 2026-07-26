"""
Session 37 — retriever.py

A tiny keyword retriever, wrapped as a LangChain Runnable so it can be
piped into an LCEL chain with the `|` operator.

We deliberately do NOT use embeddings or a vector DB here. Students built
that in Phase 3 / Phase 4. The point of this session is the LangChain
*wiring*, so we keep retrieval boring and deterministic: score each chunk
by how many question words it contains, return the top K. Swapping this
for a real Chroma retriever is a one-line change in chains.py — that is
exactly the abstraction LangChain buys you.
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from corpus import load_documents


_STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "what", "how", "of", "to",
    "for", "and", "or", "in", "on", "your", "you", "i", "we", "it", "this",
    "that", "with", "can", "me", "my", "about",
}


def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


def build_chunks() -> List[Document]:
    """Load the corpus and split it into chunks.

    Uses LangChain's RecursiveCharacterTextSplitter — the same splitter a
    production app would use — so students see the real loader+splitter
    shape even though the corpus is tiny.
    """

    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=320,
        chunk_overlap=40,
        separators=["\n\n", ". ", " "],
    )
    return splitter.split_documents(docs)


class KeywordRetriever:
    """Scores chunks by question-word overlap. Returns the top K."""

    def __init__(self, chunks: List[Document], k: int = 2) -> None:
        self.chunks = chunks
        self.k = k

    def retrieve(self, question: str) -> List[Document]:
        q_words = set(_tokens(question))
        scored = []
        for chunk in self.chunks:
            overlap = len(q_words & set(_tokens(chunk.page_content)))
            if overlap > 0:
                scored.append((overlap, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _score, chunk in scored[: self.k]]


def get_retriever(k: int = 2) -> RunnableLambda:
    """Return the retriever as a Runnable so it drops into an LCEL chain.

    The Runnable takes a question string and returns a list of Documents,
    exactly like a real LangChain retriever (`vectorstore.as_retriever()`).
    """

    retriever = KeywordRetriever(build_chunks(), k=k)
    return RunnableLambda(retriever.retrieve)


def format_docs(docs: List[Document]) -> str:
    """Join retrieved chunks into one context string with source tags.

    This is the 'stuff' step of a stuff-documents chain: every chunk is
    concatenated and handed to the prompt as one context block.
    """

    blocks = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        blocks.append(f"[{source}] {d.page_content}")
    return "\n\n".join(blocks) if blocks else "(no relevant context found)"
