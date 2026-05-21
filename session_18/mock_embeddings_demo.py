"""
Session 18 — Backup demo (no API key needed)
─────────────────────────────────────────────
If you don't have your OpenAI key working yet, this file lets you SEE the
exact same idea using `sentence-transformers` — a free, local embedding
model that runs on your laptop.

Same corpus. Same cosine search. Same 2D plot.

Install (one-time):
    $ pip install sentence-transformers scikit-learn matplotlib

Run:
    $ python mock_embeddings_demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
LOCAL_MODEL = "all-MiniLM-L6-v2"   # 384-dim, ~80 MB, runs in seconds on CPU.
TOP_K = 3
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings_plot_local.png")

LABELLED_CORPUS: list[tuple[str, str]] = [
    ("ANIMALS",  "A loyal dog wags its tail at the door."),
    ("ANIMALS",  "Puppies grow up to be the most loving pets."),
    ("ANIMALS",  "Wolves hunt together in coordinated packs."),
    ("VEHICLES", "Modern cars run on electricity instead of petrol."),
    ("VEHICLES", "Aeroplanes cross continents in just a few hours."),
    ("VEHICLES", "Trucks transport goods across long highways."),
    ("FOODS",    "Fresh pizza tastes best straight from a wood-fired oven."),
    ("FOODS",    "A homemade burger beats any fast-food chain."),
    ("FOODS",    "Spaghetti is the world's most comforting pasta dish."),
    ("EMOTIONS", "Joy spreads quickly when shared with friends."),
    ("EMOTIONS", "Fear grips the chest before a difficult conversation."),
    ("EMOTIONS", "Quiet contentment is the underrated form of happiness."),
]

TOPIC_COLOURS = {
    "ANIMALS":  "#00FF9D",
    "VEHICLES": "#FFB800",
    "FOODS":    "#FF4060",
    "EMOTIONS": "#B44FFF",
}


# ─────────────────────────────────────────────────────────────
# Cosine similarity — exactly the same formula as the API version.
# ─────────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return dot / (la * lb) if la and lb else 0.0


def search(query_vec: np.ndarray, vectors: np.ndarray,
           sentences: list[str], k: int = TOP_K) -> list[tuple[float, str]]:
    scored = [(cosine_similarity(query_vec, v), s) for v, s in zip(vectors, sentences)]
    return sorted(scored, reverse=True)[:k]


# ─────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────
def plot(coords_2d: np.ndarray, labels: list[str], sentences: list[str], path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=140)
    fig.patch.set_facecolor("#03030C")
    ax.set_facecolor("#03030C")

    for topic in TOPIC_COLOURS:
        mask = [t == topic for t in labels]
        if not any(mask):
            continue
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   s=220, color=TOPIC_COLOURS[topic],
                   edgecolors="white", linewidths=1.4, label=topic, zorder=3)

    for i, s in enumerate(sentences):
        snippet = " ".join(s.split()[:5]) + "…"
        ax.annotate(snippet, (coords_2d[i, 0], coords_2d[i, 1]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=9, color="#E8E8F0", alpha=0.95)

    ax.set_title("Meaning-Space in 2D · sentence-transformers + PCA (local)",
                 color="#00FF9D", fontsize=15, pad=18, weight="bold")
    ax.set_xlabel("PCA dim 1", color="#E8E8F0")
    ax.set_ylabel("PCA dim 2", color="#E8E8F0")
    ax.tick_params(colors="#E8E8F0")
    for s in ax.spines.values():
        s.set_color("#2A2A4A")
    ax.grid(True, color="#2A2A4A", linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    legend = ax.legend(loc="upper right", facecolor="#03030C",
                       edgecolor="#2A2A4A", labelcolor="#E8E8F0", fontsize=11)
    legend.get_frame().set_alpha(0.95)

    fig.tight_layout()
    fig.savefig(path, facecolor="#03030C", dpi=140)
    plt.close(fig)


def main() -> None:
    print("=" * 64)
    print("  SESSION 18 · BACKUP DEMO (local model, no API key)")
    print("=" * 64)

    labels    = [t for t, _ in LABELLED_CORPUS]
    sentences = [s for _, s in LABELLED_CORPUS]

    print(f"\n  [Step 1/4] Loading local model: {LOCAL_MODEL}")
    print("             First run downloads ~80 MB. Future runs use the cache.")
    model = SentenceTransformer(LOCAL_MODEL)

    print("\n  [Step 2/4] Embedding the 12 sentences...")
    vectors = model.encode(sentences, normalize_embeddings=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    print(f"             Done. Shape = {vectors.shape}   (note: 384, not 1536)")

    print("\n  [Step 3/4] Running 3 example queries...")
    queries = [
        "loyal pet that barks",
        "fast travel between countries",
        "comforting home-cooked meal",
    ]
    for q in queries:
        qv = model.encode([q])[0]
        results = search(qv, vectors, sentences)
        print(f"\n  QUERY: {q!r}")
        for rank, (score, text) in enumerate(results, start=1):
            print(f"    [{rank}]  cos = {score:+.4f}   {text}")

    print("\n  [Step 4/4] Reducing to 2D with PCA and saving plot...")
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(vectors)
    plot(coords_2d, labels, sentences, OUTPUT_FILE)
    print(f"             Saved: {OUTPUT_FILE}")

    print()
    print("  Same engine. Different model. Same idea.")
    print("  Open the PNG and look for four colour-coded clusters.")


if __name__ == "__main__":
    main()
