from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..core.base import Component
from ..core.registry import register_type
from ..core.results import EvalResult


@register_type
@dataclass(frozen=True)
class WhiteGaussianNoise(Component):
    std: float = 1.0
    mean: float = 0.0
    canonical_tags = ("statistical.white_noise", "noise")

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        del context
        values = rng.normal(loc=self.mean, scale=self.std, size=len(t))
        return EvalResult(values=values, contributions={path: values.copy()}, tags=set(self.tags()))


@register_type
@dataclass(frozen=True)
class AR1Noise(Component):
    phi: float = 0.8
    sigma: float = 0.2
    canonical_tags = ("statistical.colored_noise", "statistical.ar1", "noise")

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        del context
        n = int(len(t))
        values = np.zeros(n, dtype=float)
        if n == 0:
            return EvalResult(values=values)
        scale0 = self.sigma / np.sqrt(max(1.0 - self.phi**2, 1e-8)) if abs(self.phi) < 1 else self.sigma
        values[0] = rng.normal(scale=scale0)
        eps = rng.normal(scale=self.sigma, size=max(n - 1, 0))
        for i in range(1, n):
            values[i] = self.phi * values[i - 1] + eps[i - 1]
        return EvalResult(values=values, contributions={path: values.copy()}, tags=set(self.tags()))


@register_type
@dataclass(frozen=True)
class RandomWalkProcess(Component):
    sigma: float = 0.3
    start: float = 0.0
    canonical_tags = ("statistical.random_walk", "nonstationary")

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        del context
        increments = rng.normal(scale=self.sigma, size=len(t))
        values = self.start + np.cumsum(increments)
        return EvalResult(values=values, contributions={path: values.copy()}, tags=set(self.tags()))
