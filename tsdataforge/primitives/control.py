from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..core.base import Component
from ..core.registry import register_type
from ..core.results import EvalResult


@register_type
@dataclass(frozen=True)
class StepReference(Component):
    levels: tuple[float, ...] = (0.0, 1.0, -0.5)
    switch_points: tuple[float, ...] = (0.3, 0.65)
    canonical_tags = ("control.reference.step", "control.reference", "control", "robotics")

    def _resolve_switches(self, n: int) -> list[int]:
        if n <= 1 or not self.switch_points:
            return []
        if len(self.levels) != len(self.switch_points) + 1:
            raise ValueError("`levels` must have exactly len(switch_points) + 1 entries.")
        if max(self.switch_points) <= 1.0:
            idx = [int(round(point * (n - 1))) for point in self.switch_points]
        else:
            idx = [int(point) for point in self.switch_points]
        idx = [min(max(i, 1), n - 1) for i in idx]
        return sorted(set(idx))

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        del rng, context
        n = len(t)
        values = np.zeros(n, dtype=float)
        if n == 0:
            return EvalResult(values=values)
        switches = self._resolve_switches(n)
        starts = [0, *switches]
        ends = [*switches, n]
        for level, start, end in zip(self.levels, starts, ends):
            values[start:end] = float(level)
        states = {f"{path}/switch_points": np.asarray(switches, dtype=int)}
        return EvalResult(values=values, contributions={path: values.copy()}, states=states, tags=set(self.tags()))


@register_type
@dataclass(frozen=True)
class WaypointTrajectory(Component):
    waypoints: tuple[float, ...] = (0.0, 1.0, -0.25, 0.75)
    anchor_points: tuple[float, ...] = ()
    canonical_tags = ("control.reference.waypoint", "control.reference", "trajectory", "control", "robotics")

    def _resolve_anchor_points(self, t: np.ndarray) -> np.ndarray:
        if len(self.waypoints) == 0:
            raise ValueError("`waypoints` must contain at least one value.")
        if len(self.waypoints) == 1:
            return np.asarray([t[0] if len(t) else 0.0], dtype=float)
        if not self.anchor_points:
            if len(t) == 0:
                return np.linspace(0.0, 1.0, num=len(self.waypoints), dtype=float)
            return np.linspace(float(t[0]), float(t[-1]), num=len(self.waypoints), dtype=float)
        if len(self.anchor_points) != len(self.waypoints):
            raise ValueError("`anchor_points` must be empty or match the number of waypoints.")
        anchors = np.asarray(self.anchor_points, dtype=float)
        if len(t) > 1 and np.max(anchors) <= 1.0:
            span = float(t[-1] - t[0])
            anchors = float(t[0]) + anchors * span
        return anchors

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        del rng, context
        n = len(t)
        if n == 0:
            return EvalResult(values=np.zeros(0, dtype=float))
        anchors = self._resolve_anchor_points(t)
        order = np.argsort(anchors)
        anchors = anchors[order]
        points = np.asarray(self.waypoints, dtype=float)[order]
        values = np.interp(t, anchors, points)
        states = {f"{path}/anchor_points": anchors.copy()}
        return EvalResult(values=values, contributions={path: values.copy()}, states=states, tags=set(self.tags()))


@register_type
@dataclass(frozen=True)
class PiecewiseConstantInput(Component):
    """Random piecewise-constant multi-dimensional control input.

    Parameters
    ----------
    dim:
        Number of input channels.
    hold_min, hold_max:
        Segment length range in samples.
    scale:
        Standard deviation of each held input value.
    mean:
        Mean of each held input value.
    """

    dim: int = 2
    hold_min: int = 8
    hold_max: int = 40
    scale: float = 1.0
    mean: float = 0.0
    canonical_tags = ("control.input.piecewise_constant", "control.input", "control", "robotics")

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        del context
        n = len(t)
        dim = int(self.dim)
        if dim <= 0:
            raise ValueError("dim must be positive")
        values = np.zeros((n, dim), dtype=float)
        if n == 0:
            return EvalResult(values=values)
        hold_min = max(1, int(self.hold_min))
        hold_max = max(hold_min, int(self.hold_max))
        idx = 0
        switches: list[int] = []
        while idx < n:
            width = int(rng.integers(hold_min, hold_max + 1))
            end = min(n, idx + width)
            u = rng.normal(loc=float(self.mean), scale=abs(float(self.scale)), size=dim)
            values[idx:end, :] = u
            if end < n:
                switches.append(end)
            idx = end
        states = {
            f"{path}/switch_points": np.asarray(switches, dtype=int),
            f"{path}/dim": dim,
        }
        return EvalResult(values=values, contributions={path: values.copy()}, states=states, tags=set(self.tags()))


@register_type
@dataclass(frozen=True)
class SineInput(Component):
    """Multi-dimensional sinusoidal input (optionally with per-channel parameters)."""

    dim: int = 2
    freqs: tuple[float, ...] = (24.0,)
    amps: tuple[float, ...] = (1.0,)
    phases: tuple[float, ...] = (0.0,)
    canonical_tags = ("control.input.sine", "control.input", "control", "robotics")

    def _broadcast(self, values: tuple[float, ...], dim: int, name: str) -> np.ndarray:
        if len(values) == 1:
            return np.full(dim, float(values[0]), dtype=float)
        if len(values) != dim:
            raise ValueError(f"{name} must have length 1 or dim")
        return np.asarray(values, dtype=float)

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        del rng, context
        n = len(t)
        dim = int(self.dim)
        if dim <= 0:
            raise ValueError("dim must be positive")
        freqs = self._broadcast(self.freqs, dim, "freqs")
        amps = self._broadcast(self.amps, dim, "amps")
        phases = self._broadcast(self.phases, dim, "phases")
        if n == 0:
            return EvalResult(values=np.zeros((0, dim), dtype=float))
        values = np.zeros((n, dim), dtype=float)
        for j in range(dim):
            freq = max(float(freqs[j]), 1e-6)
            values[:, j] = float(amps[j]) * np.sin((2.0 * np.pi * t / freq) + float(phases[j]))
        states = {f"{path}/dim": dim, f"{path}/freqs": freqs, f"{path}/amps": amps, f"{path}/phases": phases}
        return EvalResult(values=values, contributions={path: values.copy()}, states=states, tags=set(self.tags()))
