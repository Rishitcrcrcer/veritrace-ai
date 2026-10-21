"""Method 1: raw attention (average of the later layers)."""
from __future__ import annotations

import numpy as np

from ..core import Explanation, ModelRun
from .base import AttributionMethod, build_explanation
from .registry import register


@register("raw_attention")
class RawAttention(AttributionMethod):
    def __init__(self, layer_start_frac: float = 0.5):
        self.layer_start_frac = layer_start_frac

    def explain(self, run: ModelRun) -> Explanation:
        n_layers = len(run.layer_attn)
        l0 = int(n_layers * self.layer_start_frac)
        matrix = np.mean(run.layer_attn[l0:], axis=0)               # (L, L)
        return build_explanation(self.name, run, matrix, {"layers": f"{l0}-{n_layers - 1}"})
