from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.signal import fftconvolve

from ..core.base import Component
from ..core.registry import register_type
from ..core.results import EvalResult


@register_type
@dataclass(frozen=True)
class BurstyPulseTrain(Component):
    base_rate: float = 0.01
    burst_probability: float = 0.05
    burst_size: int = 8
    decay: float = 4.0
    amplitude: float = 1.0
    canonical_tags = ("events.bursty", "event_based")

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        del context
        n = len(t)
        impulses = rng.binomial(1, self.base_rate, size=n).astype(float)
        burst_starts = rng.binomial(1, self.burst_probability, size=n).astype(bool)
        for idx in np.flatnonzero(burst_starts):
            end = min(n, idx + self.burst_size)
            impulses[idx:end] += self.amplitude
        kernel_t = np.arange(max(3, int(5 * self.decay)), dtype=float)
        kernel = np.exp(-kernel_t / max(self.decay, 1e-6))
        values = fftconvolve(impulses, kernel, mode="full")[:n]
        return EvalResult(
            values=values,
            contributions={path: values.copy()},
            states={f"{path}/impulses": impulses},
            tags=set(self.tags()),
        )
