"""Data containers shared by every part of the project.

Everything here is plain numpy / Python on purpose: attribution methods and
metrics can be unit-tested without a GPU or even without torch installed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ModelRun:
    """Everything one VLM call produced that attribution methods need.

    The model wrapper runs the (expensive) forward pass ONCE and fills this in.
    Every attribution method then reads from it.
    """
    answer_text: str
    tokens: list[str]               # decoded answer tokens, length A
    layer_attn: list[np.ndarray]    # one (L, L) matrix per layer, averaged over heads
    prompt_len: int                 # P: number of prompt tokens (image tokens included)
    img_pos: np.ndarray             # (n_img,) sequence positions of the image tokens
    grid: tuple[int, int]           # (gh, gw) layout of the image tokens
    image: Any = None               # PIL image (for drawing)
    question: str = ""
    model: Any = None               # wrapper object, for gradient-based methods later

    @property
    def n_answer(self) -> int:
        return len(self.tokens)

    @property
    def seq_len(self) -> int:
        return self.layer_attn[0].shape[0]


@dataclass
class Explanation:
    """The output of one attribution method for one ModelRun."""
    method: str
    heatmaps: np.ndarray        # (A, gh, gw); each token's map sums to 1 over the patches
    image_mass: np.ndarray      # (A,) attention on the image BEFORE renormalising
    tokens: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def mean_map(self) -> np.ndarray:
        return self.heatmaps.mean(axis=0)

    def entropy(self) -> np.ndarray:
        """Normalised entropy per answer token: 0 = one patch, 1 = perfectly spread out.
        NaN where the token paid (almost) no attention to the image."""
        flat = self.heatmaps.reshape(len(self.tokens), -1).astype(np.float64)
        n = flat.shape[1]
        if n < 2:
            return np.zeros(len(self.tokens))
        p = np.clip(flat, 1e-12, 1.0)
        ent = -(p * np.log(p)).sum(axis=1) / math.log(n)
        ent[self.image_mass < 1e-8] = np.nan
        return ent
