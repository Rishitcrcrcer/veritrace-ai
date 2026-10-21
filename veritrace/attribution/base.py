"""The contract every attribution method follows."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core import Explanation, ModelRun


class AttributionMethod(ABC):
    """explain(run) -> Explanation.

    The forward pass is expensive, so it is done once (ModelRun) and shared.
    Gradient-based methods can reach the model through `run.model`.
    """
    name: str = "base"

    @abstractmethod
    def explain(self, run: ModelRun) -> Explanation:
        ...


def build_explanation(name: str, run: ModelRun, matrix: np.ndarray, meta: dict | None = None) -> Explanation:
    """Turn one (L, L) token-to-token influence matrix into per-answer-token image heatmaps.

    Shared by every matrix-based method (raw attention, rollout, ...):
      1. keep the rows that DECIDED each answer token (position P-1+k decides token k)
      2. keep the image-token columns
      3. remember how much mass that was, then renormalise each row to sum to 1
      4. reshape to the patch grid
    """
    P, A = run.prompt_len, run.n_answer
    rows = np.arange(P - 1, P - 1 + A)
    block = matrix[np.ix_(rows, run.img_pos)].astype(np.float64)    # (A, n_img)
    mass = block.sum(axis=1)
    block = block / np.maximum(mass[:, None], 1e-9)

    gh, gw = run.grid
    if block.shape[1] != gh * gw:
        raise ValueError(
            f"{block.shape[1]} image tokens but grid is {gh}x{gw}. "
            "(Video input needs a per-frame grid; not supported yet.)"
        )
    return Explanation(
        method=name,
        heatmaps=block.reshape(A, gh, gw).astype(np.float32),
        image_mass=mass.astype(np.float32),
        tokens=list(run.tokens),
        meta=meta or {},
    )
