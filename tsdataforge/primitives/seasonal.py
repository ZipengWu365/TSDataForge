from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..core.base import Component
from ..core.registry import register_type
from ..core.results import EvalResult


@register_type
@dataclass(frozen=True)
class SineSeasonality(Component):
    freq: float = 24.0
    amp: float = 1.0
    phase: float = 0.0
    canonical_tags = ("periodic.single", "seasonal")

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        del rng, context
        values = self.amp * np.sin(2.0 * np.pi * t / self.freq + self.phase)
        return EvalResult(values=values, contributions={path: values.copy()}, tags=set(self.tags()))


@register_type
@dataclass(frozen=True)
class MultiSineSeasonality(Component):
    freqs: tuple[float, ...] = (24.0, 168.0)
    amps: tuple[float, ...] = (1.0, 0.5)
    phases: tuple[float, ...] = ()
    canonical_tags = ("periodic.multi", "seasonal")

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        del rng, context
        if len(self.freqs) != len(self.amps):
            raise ValueError("freqs and amps must have the same length")
        phases = self.phases if self.phases else tuple(0.0 for _ in self.freqs)
        if len(phases) != len(self.freqs):
            raise ValueError("phases must be empty or have the same length as freqs")
        values = np.zeros(len(t), dtype=float)
        for freq, amp, phase in zip(self.freqs, self.amps, phases):
            values += amp * np.sin(2.0 * np.pi * t / freq + phase)
        return EvalResult(values=values, contributions={path: values.copy()}, tags=set(self.tags()))


@register_type
@dataclass(frozen=True)
class QuasiPeriodicSeasonality(Component):
    freq: float = 24.0
    amp: float = 1.0
    phase0: float = 0.0
    jitter: float = 0.05
    canonical_tags = ("periodic.quasi", "seasonal")

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        del context
        if len(t) == 0:
            return EvalResult(values=np.zeros(0, dtype=float))
        dt = np.diff(t, prepend=t[0])
        if len(dt) > 1:
            dt[0] = np.mean(dt[1:])
        else:
            dt[0] = 1.0
        noise = rng.normal(loc=0.0, scale=self.jitter, size=len(t))
        phase = self.phase0 + np.cumsum((2.0 * np.pi / self.freq) * dt * (1.0 + noise))
        values = self.amp * np.sin(phase)
        return EvalResult(values=values, contributions={path: values.copy()}, tags=set(self.tags()))
