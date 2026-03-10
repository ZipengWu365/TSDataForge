from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from .describe import SeriesDescription, describe_series


def _iter_series(values: Any) -> list[np.ndarray]:
    """Normalize user input into a list of 1D/2D numpy arrays."""
    if isinstance(values, np.ndarray):
        arr = values
        if arr.ndim == 1:
            return [arr]
        if arr.ndim == 2:
            # (N, T) -> list of 1D
            return [arr[i, :] for i in range(arr.shape[0])]
        if arr.ndim == 3:
            # (N, T, C)
            return [arr[i, :, :] for i in range(arr.shape[0])]
        raise ValueError("values ndarray must be 1D, 2D, or 3D")
    if isinstance(values, (list, tuple)):
        out: list[np.ndarray] = []
        for v in values:
            out.append(np.asarray(v, dtype=float))
        return out
    raise TypeError("values must be a numpy array or a list/tuple of arrays")


def _iter_time(time: Any, n: int) -> list[np.ndarray | None]:
    if time is None:
        return [None] * n
    if isinstance(time, np.ndarray):
        t = np.asarray(time, dtype=float)
        if t.ndim == 1:
            # shared time
            return [t] * n
        if t.ndim == 2:
            if t.shape[0] != n:
                raise ValueError("time[0] dimension must match number of series")
            return [t[i, :] for i in range(n)]
        raise ValueError("time ndarray must be 1D or 2D")
    if isinstance(time, (list, tuple)):
        if len(time) != n:
            raise ValueError("time list length must match number of series")
        return [None if ti is None else np.asarray(ti, dtype=float) for ti in time]
    raise TypeError("time must be None, ndarray, or list/tuple")


def _summary_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "mean": float("nan"), "p75": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p75": float(np.percentile(x, 75)),
        "max": float(np.max(x)),
    }


def _signature(tags: Iterable[str]) -> str:
    core = [
        "trend",
        "seasonal",
        "ar1_like",
        "random_walk_like",
        "bursty",
        "missing",
        "irregular_sampling",
        "multivariate",
        "coupled",
    ]
    s = [t for t in core if t in set(tags)]
    return "+".join(s) if s else "none"


@dataclass
class DatasetDescription:
    n_series: int
    length_stats: dict[str, float]
    channel_stats: dict[str, float]
    missing_rate_stats: dict[str, float]
    dt_cv_stats: dict[str, float]
    tag_counts: dict[str, int] = field(default_factory=dict)
    signature_counts: dict[str, int] = field(default_factory=dict)
    per_series: list[SeriesDescription] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_dataset(
    values: Any,
    time: Any | None = None,
    *,
    max_series: int | None = 200,
    seed: int = 0,
) -> DatasetDescription:
    """Describe a dataset (collection) of time series.

    Parameters
    ----------
    values:
        - ndarray (N,T) or (N,T,C)
        - or list of arrays (variable length allowed)
    time:
        - None (assume regular)
        - 1D shared time
        - 2D per-series time
        - list of per-series time arrays

    max_series:
        For large datasets, subsample up to this many series for profiling.
    """

    series_list = _iter_series(values)
    n_total = len(series_list)
    if n_total == 0:
        raise ValueError("Empty dataset")

    # Optional subsampling
    if max_series is not None and n_total > max_series:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n_total, size=int(max_series), replace=False)
        series_list = [series_list[int(i)] for i in idx]

    n = len(series_list)
    time_list = _iter_time(time, n)

    descs: list[SeriesDescription] = []
    lengths: list[float] = []
    chans: list[float] = []
    missing_rates: list[float] = []
    dt_cvs: list[float] = []
    tag_counts: dict[str, int] = {}
    sig_counts: dict[str, int] = {}

    for v, t in zip(series_list, time_list):
        d = describe_series(v, t)
        descs.append(d)
        lengths.append(float(d.length))
        chans.append(float(d.n_channels))
        missing_rates.append(float(d.missing_rate))
        dt_cvs.append(float(d.dt_cv))
        for tag in d.inferred_tags:
            tag_counts[tag] = int(tag_counts.get(tag, 0) + 1)
        sig = _signature(d.inferred_tags)
        sig_counts[sig] = int(sig_counts.get(sig, 0) + 1)

    return DatasetDescription(
        n_series=n_total,
        length_stats=_summary_stats(np.asarray(lengths, dtype=float)),
        channel_stats=_summary_stats(np.asarray(chans, dtype=float)),
        missing_rate_stats=_summary_stats(np.asarray(missing_rates, dtype=float)),
        dt_cv_stats=_summary_stats(np.asarray(dt_cvs, dtype=float)),
        tag_counts=dict(sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        signature_counts=dict(sorted(sig_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        per_series=descs,
    )
