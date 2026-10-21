"""Method 2: attention rollout (Abnar & Zuidema, 2020).

Chains the attention matrices of all layers so the result says how much each
INPUT token's information can reach each output position through the whole stack,
not just through the last layer.
"""
from __future__ import annotations

import numpy as np

from ..core import Explanation, ModelRun
from .base import AttributionMethod, build_explanation
from .registry import register


@register("rollout")
class AttentionRollout(AttributionMethod):
    def __init__(self, start_layer: int = 0, residual_weight: float = 0.5):
        """residual_weight: share given to the identity (skip connection) in every layer.
        0.5 is the value from the paper; 0.0 means 'pure attention, no skip path'."""
        self.start_layer = start_layer
        self.residual_weight = residual_weight

    def explain(self, run: ModelRun) -> Explanation:
        L = run.seq_len
        eye = np.eye(L)
        w = self.residual_weight

        rollout = eye.copy()
        for attn in run.layer_attn[self.start_layer:]:
            a_hat = (1.0 - w) * attn.astype(np.float64) + w * eye   # add the residual path
            a_hat /= a_hat.sum(axis=-1, keepdims=True)              # keep rows summing to 1
            rollout = a_hat @ rollout                               # later layer on the LEFT
        return build_explanation(
            self.name, run, rollout,
            {"start_layer": self.start_layer, "residual_weight": w},
        )
