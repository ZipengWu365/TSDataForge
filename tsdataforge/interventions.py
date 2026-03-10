from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .core.base import Serializable
from .core.registry import register_type


@register_type
@dataclass(frozen=True)
class InterventionSpec(Serializable):
    """Serializable intervention definition for control/causal generators.

    Parameters
    ----------
    target:
        Semantic target for the intervention. Supported values depend on the
        component, but common choices are ``"variable"``, ``"input"``,
        ``"state"``, ``"output"``, and ``"treatment"``.
    index:
        Target channel / variable index within the target group.
    start, end:
        Start / end time specified either as integer indices or fractions in
        ``[0, 1]`` of the episode length. If ``end`` is ``None``, the
        intervention persists until the end of the episode.
    value:
        Scalar or sequence of values applied while the mask is active.
    mode:
        ``"override"`` (default) or ``"add"``.
    """

    target: str = "variable"
    index: int = 0
    start: int | float = 0
    end: int | float | None = None
    value: float | Sequence[float] = 0.0
    mode: str = "override"
    name: str | None = None

    def label(self) -> str:
        return self.name or f"{self.target}_{self.index}_{self.mode}"

    def active_mask(self, n_steps: int) -> np.ndarray:
        return intervention_mask(n_steps, self.start, self.end)

    def values_over_time(self, n_steps: int) -> np.ndarray:
        mask = self.active_mask(n_steps)
        _, values = intervention_mask_values(n_steps, self, precomputed_mask=mask)
        return values


InterventionLike = InterventionSpec | Mapping[str, Any]


def _coerce_bound(bound: int | float | None, n_steps: int) -> int | None:
    if bound is None:
        return None
    if isinstance(bound, (int, np.integer)):
        return int(bound)
    x = float(bound)
    if 0.0 <= x <= 1.0 and n_steps > 1:
        return int(round(x * (n_steps - 1)))
    return int(round(x))



def intervention_mask(n_steps: int, start: int | float, end: int | float | None = None) -> np.ndarray:
    n = int(max(0, n_steps))
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    s = _coerce_bound(start, n)
    e = _coerce_bound(end, n)
    if s is None:
        s = 0
    s = int(np.clip(s, 0, n - 1))
    if e is None:
        e = n
    else:
        e = int(np.clip(e, 0, n))
        if e <= s:
            e = min(n, s + 1)
    mask[s:e] = True
    return mask



def intervention_mask_values(
    n_steps: int,
    intervention: InterventionLike,
    *,
    precomputed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a boolean mask and per-step values for an intervention."""

    n = int(max(0, n_steps))
    if isinstance(intervention, InterventionSpec):
        spec = intervention
    elif isinstance(intervention, Mapping):
        if "at" in intervention:
            mask = np.zeros(n, dtype=bool)
            values = np.zeros(n, dtype=float)
            at = intervention.get("at", ())
            at_list = list(at) if isinstance(at, (list, tuple, np.ndarray)) else [at]
            if at_list:
                if max(float(a) for a in at_list) <= 1.0:
                    idx = [int(round(float(a) * max(n - 1, 0))) for a in at_list]
                else:
                    idx = [int(round(float(a))) for a in at_list]
                idx = [i for i in idx if 0 <= i < n]
                value = intervention.get("value", 0.0)
                if isinstance(value, (list, tuple, np.ndarray)):
                    vals = list(value)
                    if len(vals) == 1 and len(idx) > 1:
                        vals = vals * len(idx)
                    if len(vals) != len(idx):
                        raise ValueError("Intervention 'value' length must match 'at' length")
                else:
                    vals = [float(value)] * len(idx)
                for i, v in zip(idx, vals):
                    mask[int(i)] = True
                    values[int(i)] = float(v)
            return mask, values
        spec = InterventionSpec(
            target=str(intervention.get("target", intervention.get("kind", "variable"))),
            index=int(intervention.get("index", intervention.get("var", 0))),
            start=intervention.get("start", intervention.get("at", 0)),
            end=intervention.get("end"),
            value=intervention.get("value", 0.0),
            mode=str(intervention.get("mode", "override")),
            name=intervention.get("name"),
        )
    else:
        raise TypeError(f"Unsupported intervention type: {type(intervention)!r}")

    mask = precomputed_mask if precomputed_mask is not None else spec.active_mask(n)
    values = np.zeros(n, dtype=float)
    value = spec.value
    active = int(mask.sum())
    if active == 0:
        return mask.astype(bool), values
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 1:
            values[mask] = float(arr[0])
        elif arr.size == active:
            values[mask] = arr
        elif arr.size == n:
            values = arr.astype(float, copy=False)
            values[~mask] = 0.0
        else:
            raise ValueError(
                "Intervention value sequence must have length 1, n_steps, or number of active steps"
            )
    else:
        values[mask] = float(value)
    return mask.astype(bool), values



def merge_interventions(*groups: Sequence[InterventionLike] | None) -> tuple[InterventionLike, ...]:
    out: list[InterventionLike] = []
    for group in groups:
        if not group:
            continue
        out.extend(list(group))
    return tuple(out)
