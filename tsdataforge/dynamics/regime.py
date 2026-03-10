from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..core.base import Component
from ..core.registry import register_type
from ..core.results import EvalResult


@register_type
@dataclass(frozen=True)
class RegimeSwitch(Component):
    regimes: tuple[Component, ...]
    transition_matrix: np.ndarray | None = None
    initial_probs: tuple[float, ...] | None = None
    canonical_tags = ("dynamic.regime_switching", "regime_switching")

    def children(self) -> tuple[Component, ...]:
        return self.regimes

    def _default_transition(self, n_regimes: int) -> np.ndarray:
        stay = 0.96
        if n_regimes == 1:
            return np.array([[1.0]], dtype=float)
        move = (1.0 - stay) / (n_regimes - 1)
        matrix = np.full((n_regimes, n_regimes), move, dtype=float)
        np.fill_diagonal(matrix, stay)
        return matrix

    def _sample_states(self, n_steps: int, rng: np.random.Generator) -> np.ndarray:
        n_regimes = len(self.regimes)
        transition = self.transition_matrix if self.transition_matrix is not None else self._default_transition(n_regimes)
        transition = np.asarray(transition, dtype=float)
        if transition.shape != (n_regimes, n_regimes):
            raise ValueError("transition_matrix must have shape (n_regimes, n_regimes)")
        probs = np.asarray(self.initial_probs if self.initial_probs is not None else np.full(n_regimes, 1.0 / n_regimes), dtype=float)
        probs = probs / probs.sum()
        states = np.zeros(n_steps, dtype=int)
        states[0] = int(rng.choice(n_regimes, p=probs))
        for i in range(1, n_steps):
            states[i] = int(rng.choice(n_regimes, p=transition[states[i - 1]]))
        return states

    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, object],
        path: str,
    ) -> EvalResult:
        n = len(t)
        if n == 0:
            return EvalResult(values=np.zeros(0, dtype=float))
        regime_states = self._sample_states(n, rng)
        values = np.zeros(n, dtype=float)
        contributions: dict[str, np.ndarray] = {}
        states: dict[str, object] = {}
        tags: set[str] = set(self.tags())
        for idx, regime in enumerate(self.regimes):
            child_path = f"{path}/regime_{idx}:{regime.label()}"
            result = regime.evaluate(t, rng, context, path=child_path)
            active = regime_states == idx
            values[active] = result.values[active]
            for key, arr in result.contributions.items():
                masked = np.zeros_like(arr)
                masked[active] = arr[active]
                contributions[key] = masked
            states.update(result.states)
            tags.update(result.tags)
        changepoints = np.flatnonzero(np.diff(regime_states, prepend=regime_states[0]))
        states[f"{path}/regime_state"] = regime_states
        states[f"{path}/changepoints"] = changepoints.astype(int)
        return EvalResult(values=values, contributions=contributions, states=states, tags=tags)
