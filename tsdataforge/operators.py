from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.signal import fftconvolve

from .core.base import Component
from .core.registry import register_type
from .core.results import EvalResult


def _merge_results(results: list[EvalResult]) -> tuple[dict[str, np.ndarray], dict[str, object], set[str]]:
    contributions: dict[str, np.ndarray] = {}
    states: dict[str, object] = {}
    tags: set[str] = set()
    for result in results:
        contributions.update(result.contributions)
        states.update(result.states)
        tags.update(result.tags)
    return contributions, states, tags


@register_type
@dataclass(frozen=True)
class Add(Component):
    components: tuple[Component, ...]
    canonical_tags = ("composition.add",)

    def children(self) -> tuple[Component, ...]:
        return self.components

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        child_results = [
            component.evaluate(t, rng, context, path=f"{path}/{idx}:{component.label()}")
            for idx, component in enumerate(self.components)
        ]
        values = np.sum([result.values for result in child_results], axis=0)
        contributions, states, tags = _merge_results(child_results)
        tags.update(self.tags())
        return EvalResult(values=values, contributions=contributions, states=states, tags=tags)


@register_type
@dataclass(frozen=True)
class Multiply(Component):
    components: tuple[Component, ...]
    canonical_tags = ("composition.multiply",)

    def children(self) -> tuple[Component, ...]:
        return self.components

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        child_results = [
            component.evaluate(t, rng, context, path=f"{path}/{idx}:{component.label()}")
            for idx, component in enumerate(self.components)
        ]
        values = np.ones(len(t), dtype=float)
        for result in child_results:
            values *= result.values
        contributions, states, tags = _merge_results(child_results)
        tags.update(self.tags())
        return EvalResult(values=values, contributions=contributions, states=states, tags=tags)


@register_type
@dataclass(frozen=True)
class Convolve(Component):
    component: Component
    kernel: tuple[float, ...]
    canonical_tags = ("composition.convolve",)

    def children(self) -> tuple[Component, ...]:
        return (self.component,)

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        child_path = f"{path}/source:{self.component.label()}"
        result = self.component.evaluate(t, rng, context, path=child_path)
        kernel = np.asarray(self.kernel, dtype=float)
        src = np.asarray(result.values, dtype=float)
        if src.ndim == 1:
            values = fftconvolve(src, kernel, mode="full")[: len(t)]
        elif src.ndim == 2:
            values = np.vstack([fftconvolve(src[:, j], kernel, mode="full")[: len(t)] for j in range(src.shape[1])]).T
        else:
            raise ValueError("Convolve only supports 1D or 2D inputs.")
        states = dict(result.states)
        states[f"{path}/kernel"] = kernel
        tags = set(result.tags)
        tags.update(self.tags())
        return EvalResult(values=values, contributions=result.contributions, states=states, tags=tags)


@register_type
@dataclass(frozen=True)
class TimeWarp(Component):
    component: Component
    strength: float = 0.1
    canonical_tags = ("composition.time_warp",)

    def children(self) -> tuple[Component, ...]:
        return (self.component,)

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        if len(t) == 0:
            return EvalResult(values=np.zeros(0, dtype=float))
        dt = np.diff(t, prepend=t[0])
        if len(dt) > 1:
            dt[0] = np.mean(dt[1:])
        else:
            dt[0] = 1.0
        increments = np.clip(dt * (1.0 + rng.normal(scale=self.strength, size=len(t))), 1e-6, None)
        warped = np.cumsum(increments)
        warped -= warped[0]
        total_span = float(t[-1] - t[0]) if len(t) > 1 else 1.0
        if warped[-1] > 0:
            warped *= total_span / warped[-1]
        warped += t[0]
        result = self.component.evaluate(warped, rng, context, path=f"{path}/warped:{self.component.label()}")
        states = dict(result.states)
        states[f"{path}/warp_time"] = warped
        tags = set(result.tags)
        tags.update(self.tags())
        return EvalResult(values=result.values, contributions=result.contributions, states=states, tags=tags)


@register_type
@dataclass(frozen=True)
class Stack(Component):
    """Concatenate child component outputs along the channel dimension.

    This operator is the recommended way to build multivariate series from
    multiple scalar components (or to concatenate multivariate components).

    Rules
    -----
    - 1D child output is treated as shape (T, 1)
    - 2D child output is treated as shape (T, C)
    - Outputs are concatenated along axis=1 resulting in shape (T, sum(C))
    """

    components: tuple[Component, ...]
    canonical_tags = ("composition.stack", "multivariate")

    def children(self) -> tuple[Component, ...]:
        return self.components

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        child_results = [
            component.evaluate(t, rng, context, path=f"{path}/{idx}:{component.label()}")
            for idx, component in enumerate(self.components)
        ]
        blocks: list[np.ndarray] = []
        for result in child_results:
            arr = np.asarray(result.values, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.ndim == 2:
                pass
            else:
                raise ValueError("Stack only supports 1D or 2D child outputs.")
            blocks.append(arr)
        values = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(t), 0), dtype=float)
        contributions, states, tags = _merge_results(child_results)
        states[f"{path}/n_channels"] = int(values.shape[1])
        tags.update(self.tags())
        return EvalResult(values=values, contributions=contributions, states=states, tags=tags)
