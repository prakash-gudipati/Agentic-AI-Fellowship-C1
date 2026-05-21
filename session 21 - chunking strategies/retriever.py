
from dataclasses import dataclass
from typing import List

import numpy as np

from chunkers.base import Chunk
from embeddings import embed_texts, embed_one


@dataclass
class Retrieved:
    """One retrieved chunk plus its similarity score and source."""
    chunk: Chunk
    score: float

    def __repr__(self) -> str:
        preview = self.chunk.text[:60].replace("\n", " ")
        return f"Retrieved(score={self.score:.3f}, source={self.chunk.source!r}, text={preview!r})"


class InMemoryStore:
    """A list of chunks + a parallel matrix of their embedding vectors.

    Purposely tiny. No persistence, no metadata filtering, no ANN index.
    Sessions 21–28 progressively replace this with a real vector DB.
    """

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.vectors: np.ndarray = np.zeros((0, 1536), dtype=np.float32)

    def add(self, chunks: List[Chunk]) -> None:
        """Embed `chunks` and append them to the store."""
        if not chunks:
            return
        new_vectors = embed_texts([c.text for c in chunks])
        # Normalise vectors so cosine similarity reduces to a dot product.
        new_vectors = _l2_normalise(new_vectors)
        self.chunks.extend(chunks)
        if self.vectors.shape[0] == 0:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
        print(f"[retriever]  store now holds {len(self.chunks)} chunks")

    def search(self, query: str, k: int = 3) -> List[Retrieved]:
        """Embed the query and return the top-K most-similar chunks."""
        if len(self.chunks) == 0:
            return []
        q_vec = embed_one(query)
        q_vec = _l2_normalise(q_vec.reshape(1, -1))[0]

        # Cosine similarity == dot product when both sides are L2-normalised.
        scores = self.vectors @ q_vec

        top_indices = np.argsort(-scores)[:k]
        results = [Retrieved(chunk=self.chunks[i], score=float(scores[i]))
                   for i in top_indices]

        print(f"[retriever]  query={query!r}  top-{k} scores: "
              + ", ".join(f"{r.score:.3f}" for r in results))
        return results


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Divide each row by its L2 norm so dot product == cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


if __name__ == "__main__":
    # Quick smoke test — run python retriever.py
    from chunkers import FixedCharChunker
    chunker = FixedCharChunker()
    docs = [
        ("rag.txt", "RAG stands for Retrieval-Augmented Generation. " * 6),
        ("manual.txt", "Press the power button for three seconds to install. " * 6),
    ]
    store = InMemoryStore()
    chunks = []
    for src, text in docs:
        chunks.extend(chunker.split(text, source=src, start_id=len(chunks)))
    store.add(chunks)

    for q in ["What is RAG?", "How do I install the device?"]:
        print()
        results = store.search(q, k=2)
        for r in results:
            print(" ", r)
