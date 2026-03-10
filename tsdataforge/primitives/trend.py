from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..core.base import Component
from ..core.registry import register_type
from ..core.results import EvalResult


@register_type
@dataclass(frozen=True)
class LinearTrend(Component):
    slope: float = 0.01
    intercept: float = 0.0
    canonical_tags = ("trend.linear", "trend")

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        del rng, context
        centered = t - float(t[0])
        values = self.intercept + self.slope * centered
        return EvalResult(values=values, contributions={path: values.copy()}, tags=set(self.tags()))


@register_type
@dataclass(frozen=True)
class PiecewiseLinearTrend(Component):
    knots: tuple[float, ...] = (0.4, 0.7)
    slopes: tuple[float, ...] = (0.01, -0.005, 0.02)
    intercept: float = 0.0
    canonical_tags = ("trend.piecewise_linear", "trend")

    def _resolve_knots(self, n: int) -> list[int]:
        if not self.knots:
            return []
        knots = list(self.knots)
        if max(knots) <= 1.0:
            idx = [int(round(k * (n - 1))) for k in knots]
        else:
            idx = [int(k) for k in knots]
        idx = [min(max(i, 1), n - 1) for i in idx]
        idx = sorted(set(idx))
        return idx

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        del rng, context
        n = len(t)
        breaks = self._resolve_knots(n)
        if len(self.slopes) != len(breaks) + 1:
            raise ValueError("`slopes` must have exactly len(knots) + 1 segments.")
        values = np.zeros(n, dtype=float)
        segment_starts = [0, *breaks]
        segment_ends = [*breaks, n]
        current_value = self.intercept
        for seg_id, (start, end) in enumerate(zip(segment_starts, segment_ends)):
            local_t = t[start:end] - t[start]
            values[start:end] = current_value + self.slopes[seg_id] * local_t
            if end > start:
                current_value = values[end - 1]
        return EvalResult(values=values, contributions={path: values.copy()}, tags=set(self.tags()))
