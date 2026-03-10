from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .core.base import Serializable
from .core.registry import register_type


class Policy(Serializable, ABC):
    """Serializable action policy used by control / counterfactual generators."""

    def label(self) -> str:
        return getattr(self, "name", None) or self.__class__.__name__.lower()

    @abstractmethod
    def action(
        self,
        *,
        state: np.ndarray,
        output: np.ndarray,
        t_index: int,
        time_value: float,
        prev_action: np.ndarray,
        context: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError



def _as_1d(x: np.ndarray | Sequence[float] | float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)



def _pick_source(source: str, *, state: np.ndarray, output: np.ndarray) -> np.ndarray:
    key = str(source).lower().strip()
    if key in {"state", "x"}:
        return _as_1d(state)
    if key in {"output", "y", "observation", "obs"}:
        return _as_1d(output)
    raise ValueError(f"Unsupported policy source: {source!r}")


@register_type
@dataclass(frozen=True)
class ConstantPolicy(Policy):
    action_value: Sequence[float] | float
    name: str | None = None

    def action(self, *, state: np.ndarray, output: np.ndarray, t_index: int, time_value: float, prev_action: np.ndarray, context: Mapping[str, Any] | None = None) -> np.ndarray:
        del state, output, t_index, time_value, prev_action, context
        return _as_1d(self.action_value)


@register_type
@dataclass(frozen=True)
class PiecewiseConstantPolicy(Policy):
    actions: tuple[Sequence[float] | float, ...]
    switch_points: tuple[int | float, ...] = ()
    name: str | None = None

    def action(self, *, state: np.ndarray, output: np.ndarray, t_index: int, time_value: float, prev_action: np.ndarray, context: Mapping[str, Any] | None = None) -> np.ndarray:
        del state, output, prev_action, context
        idx = 0
        for sp in self.switch_points:
            boundary = float(sp)
            cur = float(time_value) if abs(boundary) <= 1.0 else float(t_index)
            if cur >= boundary:
                idx += 1
        idx = min(idx, max(0, len(self.actions) - 1))
        return _as_1d(self.actions[idx])


@register_type
@dataclass(frozen=True)
class LinearFeedbackPolicy(Policy):
    K: np.ndarray
    bias: Sequence[float] | float | None = None
    source: str = "state"
    negate: bool = True
    name: str | None = None

    def action(self, *, state: np.ndarray, output: np.ndarray, t_index: int, time_value: float, prev_action: np.ndarray, context: Mapping[str, Any] | None = None) -> np.ndarray:
        del t_index, time_value, prev_action, context
        obs = _pick_source(self.source, state=state, output=output)
        K = np.asarray(self.K, dtype=float)
        if K.ndim == 1:
            K = K.reshape(1, -1)
        if K.shape[1] != obs.size:
            raise ValueError(f"LinearFeedbackPolicy expected obs dim {K.shape[1]}, got {obs.size}")
        out = K @ obs
        if self.negate:
            out = -out
        if self.bias is not None:
            out = out + _as_1d(self.bias)
        return _as_1d(out)


@register_type
@dataclass(frozen=True)
class ThresholdPolicy(Policy):
    index: int = 0
    threshold: float = 0.0
    low_action: Sequence[float] | float = 0.0
    high_action: Sequence[float] | float = 1.0
    source: str = "state"
    name: str | None = None

    def action(self, *, state: np.ndarray, output: np.ndarray, t_index: int, time_value: float, prev_action: np.ndarray, context: Mapping[str, Any] | None = None) -> np.ndarray:
        del t_index, time_value, prev_action, context
        obs = _pick_source(self.source, state=state, output=output)
        if not (0 <= int(self.index) < obs.size):
            raise ValueError(f"ThresholdPolicy index out of bounds: {self.index}")
        if float(obs[int(self.index)]) >= float(self.threshold):
            return _as_1d(self.high_action)
        return _as_1d(self.low_action)
