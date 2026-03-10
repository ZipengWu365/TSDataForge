from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..series import GeneratedSeries
from ..specs import SeriesSpec
from ..trace import SeriesTrace


@dataclass
class ExternalRollout:
    """A minimal container for data produced by an external simulator.

    This is intentionally generic so that MuJoCo / IsaacGym / PyBullet / Gymnasium
    users can adapt without TSDataForge taking hard dependencies.
    """

    time: np.ndarray
    values: np.ndarray
    channel_names: list[str] | None = None
    meta: dict[str, Any] | None = None


class SimulatorAdapter(Protocol):
    """Protocol for external simulator adapters.

    Implementations should return an `ExternalRollout` containing at least
    `time` and `values`.
    """

    def rollout(self, *, seed: int | None = None, **kwargs: Any) -> ExternalRollout: ...


def wrap_external_series(
    values: np.ndarray,
    time: np.ndarray | None = None,
    *,
    channel_names: list[str] | None = None,
    name: str = "external",
    tags: tuple[str, ...] = ("external",),
    meta: dict[str, Any] | None = None,
) -> GeneratedSeries:
    """Wrap a real/simulated external time series as a `GeneratedSeries`.

    The returned object can be used with TSDataForge analysis/reporting tools.

    Notes
    -----
    - `spec` is set to a lightweight placeholder (structure_id="external").
    - `trace.latent` is set equal to the observed values.
    - `trace.masks["observed_mask"]` stores finite-value mask.
    """

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        y = arr
    elif arr.ndim == 2:
        y = arr
    else:
        raise ValueError("values must be 1D or 2D")

    if time is None:
        t = np.arange(arr.shape[0], dtype=float)
    else:
        t = np.asarray(time, dtype=float)
        if t.shape[0] != arr.shape[0]:
            raise ValueError("time length must match values length")

    # Placeholder spec
    spec = SeriesSpec(latent=None, structure_id="external", tags=tags, name=name)

    # Trace with masks for analysis/reporting
    obs_mask = np.isfinite(arr)
    trace = SeriesTrace(
        seed=None,
        spec={"source": "external", "name": name, "meta": meta or {}},
        structure_id="external",
        time=t,
        observed_time=t,
        latent=arr,
        observed=arr,
        contributions={},
        states={"external/channel_names": channel_names or []},
        masks={"observed_mask": obs_mask},
        tags=tags,
    )

    return GeneratedSeries(time=t, values=arr, spec=spec, trace=trace)
