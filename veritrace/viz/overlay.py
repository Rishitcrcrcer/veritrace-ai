"""Heatmap drawing. Pure numpy / matplotlib / PIL."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..core import Explanation, ModelRun


def to_heatmap(grid_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(gh, gw) -> (H, W) in [0, 1]. Min-max scaled and smoothly upsampled: DISPLAY ONLY.
    Min-max makes weak maps look vivid, so read the entropy numbers too."""
    m = grid_map - grid_map.min()
    m = m / (m.max() + 1e-9)
    img = Image.fromarray((m * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    return np.asarray(img) / 255.0


def pick_tokens(tokens: list[str], word: str | None = None, k: int = 5) -> list[int]:
    """Indices of tokens to draw: those containing `word`, else the first k word-like tokens."""
    if word:
        hits = [i for i, t in enumerate(tokens) if word.lower() in t.lower()]
        if hits:
            return hits[:k]
    wordlike = [i for i, t in enumerate(tokens) if any(c.isalpha() for c in t)]
    return wordlike[:k]


def save_comparison(run: ModelRun, explanations: list[Explanation], path: str,
                    token_ids: list[int]) -> None:
    """One row per method: input | mean over answer | one column per chosen token."""
    n_rows, n_cols = len(explanations), 2 + len(token_ids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows), squeeze=False)
    size = run.image.size

    for r, ex in enumerate(explanations):
        ent = ex.entropy()

        ax = axes[r][0]
        ax.imshow(run.image)
        ax.axis("off")
        ax.text(-0.04, 0.5, ex.method, rotation=90, transform=ax.transAxes,
                ha="right", va="center", fontsize=10)
        if r == 0:
            ax.set_title("input", fontsize=8)

        ax = axes[r][1]
        ax.imshow(run.image)
        ax.imshow(to_heatmap(ex.mean_map, size), cmap="jet", alpha=0.45)
        ax.set_title(f"mean\nH={np.nanmean(ent):.2f}", fontsize=8)
        ax.axis("off")

        for c, ti in enumerate(token_ids):
            ax = axes[r][c + 2]
            ax.imshow(run.image)
            ax.imshow(to_heatmap(ex.heatmaps[ti], size), cmap="jet", alpha=0.45)
            ax.set_title(f"{ex.tokens[ti]!r}\nH={ent[ti]:.2f}  img={ex.image_mass[ti]:.2f}",
                         fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
