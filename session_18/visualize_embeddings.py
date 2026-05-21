"""
Session 18 — Part D: Visualise Meaning-Space in 2D
──────────────────────────────────────────────────
Goal: Reduce 1,536-dimensional embeddings down to 2D using PCA, then plot
them with matplotlib. Watch sentences from the same topic cluster together.

This is the visual moment of Phase 3: students SEE that meaning has geography.

Run this file:
    $ python visualize_embeddings.py

Output:
    embeddings_plot.png   (saved next to this file)

PROD PATTERN — same embedding model as similarity_search.py.
PROD PATTERN — pipeline-style logging with [Step N] prefixes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings_plot.png")

# Same corpus structure as similarity_search.py — but here we keep the
# topic label alongside each sentence so we can colour-code the plot.
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

# Topic → colour. Matches the slide deck colours so students can connect
# what they saw in the slides to what they see in matplotlib.
TOPIC_COLOURS: dict[str, str] = {
    "ANIMALS":  "#00FF9D",   # mint
    "VEHICLES": "#FFB800",   # amber
    "FOODS":    "#FF4060",   # red
    "EMOTIONS": "#B44FFF",   # purple
}

PLOT_BG    = "#03030C"
PLOT_FG    = "#E8E8F0"
GRID_COLOR = "#2A2A4A"


# ─────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Add it to .env.")
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed a list of strings and return shape (n, 1536) numpy array."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# 2D projection
# ─────────────────────────────────────────────────────────────
def reduce_to_2d(vectors: np.ndarray) -> np.ndarray:
    """Use PCA to squash a (n, 1536) matrix down to (n, 2) for plotting.

    PCA = Principal Component Analysis. It finds the two directions in the
    1,536-D space along which the data varies the MOST, and projects the
    vectors onto those two directions. We lose information — a lot of it —
    but enough structure survives that clusters become visible to the eye.
    """
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(vectors)


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────
def plot_embeddings(coords_2d: np.ndarray,
                    labels: list[str],
                    sentences: list[str],
                    output_path: str) -> None:
    """Render a dark-themed scatter plot, save to disk."""
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=140)
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)

    # Plot each topic separately so we can use distinct colours.
    for topic in TOPIC_COLOURS:
        mask = [t == topic for t in labels]
        if not any(mask):
            continue
        xs = coords_2d[mask, 0]
        ys = coords_2d[mask, 1]
        ax.scatter(xs, ys, s=220, color=TOPIC_COLOURS[topic],
                   edgecolors="white", linewidths=1.4,
                   label=topic, zorder=3)

    # Annotate each point with a TINY snippet of the sentence.
    for i, sentence in enumerate(sentences):
        snippet = sentence.split()[:5]
        ax.annotate(
            " ".join(snippet) + "…",
            xy=(coords_2d[i, 0], coords_2d[i, 1]),
            xytext=(8, 6), textcoords="offset points",
            fontsize=9, color=PLOT_FG, alpha=0.95,
        )

    # Axis cosmetics — match the slide deck's dark theme.
    ax.set_title("Meaning-Space in 2D · text-embedding-3-small + PCA",
                 color="#00FF9D", fontsize=15, pad=18, weight="bold")
    ax.set_xlabel("PCA dimension 1", color=PLOT_FG, fontsize=12)
    ax.set_ylabel("PCA dimension 2", color=PLOT_FG, fontsize=12)
    ax.tick_params(colors=PLOT_FG)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)

    legend = ax.legend(loc="upper right", facecolor=PLOT_BG,
                       edgecolor=GRID_COLOR, labelcolor=PLOT_FG, fontsize=11)
    legend.get_frame().set_alpha(0.95)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=PLOT_BG, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 64)
    print("  SESSION 18 · PART D — VISUALISE MEANING-SPACE IN 2D")
    print("=" * 64)

    labels    = [topic    for topic, _ in LABELLED_CORPUS]
    sentences = [sentence for _, sentence in LABELLED_CORPUS]

    print("\n  [Step 1/3] Embedding 12 sentences across 4 topics...")
    client = get_client()
    vectors = embed_texts(client, sentences)
    print(f"             Done. Shape = {vectors.shape}")

    print("\n  [Step 2/3] Reducing 1,536-D vectors to 2-D with PCA...")
    coords_2d = reduce_to_2d(vectors)
    print(f"             Done. Shape = {coords_2d.shape}")

    print(f"\n  [Step 3/3] Rendering scatter plot to {OUTPUT_FILE} ...")
    plot_embeddings(coords_2d, labels, sentences, OUTPUT_FILE)
    print("             Done.")

    # Quick numeric report — sanity check that clusters survived the squash.
    print("\n  Cluster centres (mean of each topic in 2D):")
    for topic in TOPIC_COLOURS:
        mask = np.array([t == topic for t in labels])
        if mask.any():
            cx, cy = coords_2d[mask].mean(axis=0)
            print(f"    {topic:9s}  ({cx:+.3f}, {cy:+.3f})")

    print()
    print(f"  Open the PNG: {OUTPUT_FILE}")
    print("  You should see four colour-coded clusters, one per topic.")
    print("  That is meaning-space. That is the geometry of language.")


if __name__ == "__main__":
    main()
