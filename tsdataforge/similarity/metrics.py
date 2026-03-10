from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import exp
from typing import Any, Mapping

import numpy as np
from scipy.signal import periodogram

from ..analysis.describe import describe_series
from ..series import GeneratedSeries


ArrayLike = Any


@dataclass
class SimilarityResult:
    """Similarity report for two time series.

    The result is intentionally explainable: it records the preprocessing
    choices, per-metric scores, aggregate score, and lightweight evidence such
    as detected tag overlap and dominant transforms.
    """

    reference_name: str
    candidate_name: str
    aggregate_score: float
    metrics: dict[str, float] = field(default_factory=dict)
    transform: str = "zscore"
    target_length: int = 256
    tag_overlap: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Similarity: {self.reference_name} vs {self.candidate_name}",
            "",
            f"- Aggregate score: **{self.aggregate_score:.3f}**",
            f"- Transform: `{self.transform}`",
            f"- Target length: `{self.target_length}`",
            f"- Tag overlap: {', '.join(self.tag_overlap) if self.tag_overlap else 'none' }",
            "",
            "## Metrics",
            "",
        ]
        for key, value in sorted(self.metrics.items()):
            lines.append(f"- `{key}`: {value:.3f}")
        if self.summary:
            lines.extend(["", "## Interpretation", "", self.summary])
        return "\n".join(lines).strip() + "\n"


@dataclass
class SimilarityMatrix:
    """Pairwise similarity matrix for a named collection of series."""

    names: list[str]
    matrix: np.ndarray
    metric: str = "aggregate"
    transform: str = "zscore"

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "matrix": np.asarray(self.matrix, dtype=float).tolist(),
            "metric": self.metric,
            "transform": self.transform,
        }

    def to_markdown(self) -> str:
        headers = ["series", *self.names]
        lines = [
            f"# Pairwise similarity ({self.metric})",
            "",
            f"Transform: `{self.transform}`",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        arr = np.asarray(self.matrix, dtype=float)
        for i, name in enumerate(self.names):
            row = [name, *[f"{float(arr[i, j]):.3f}" for j in range(arr.shape[1])]]
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines).strip() + "\n"


def _coerce_series(series: ArrayLike, time: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    if isinstance(series, GeneratedSeries):
        values = np.asarray(series.values, dtype=float)
        if values.ndim == 2:
            values = values[:, 0]
        t = np.asarray(series.time if time is None else time, dtype=float)
        name = str(series.spec.name or series.spec.structure_id or "series")
        return values.reshape(-1), t.reshape(-1), name
    values = np.asarray(series, dtype=float)
    if values.ndim == 2:
        values = values[:, 0]
    elif values.ndim != 1:
        raise ValueError("series must be 1D, 2D, or GeneratedSeries")
    t = np.arange(values.shape[0], dtype=float) if time is None else np.asarray(time, dtype=float).reshape(-1)
    if t.shape[0] != values.shape[0]:
        raise ValueError("time length must match series length")
    return values.reshape(-1), t, "series"



def _fill_nan(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr)
    if mask.all():
        return arr
    if not mask.any():
        return np.zeros_like(arr, dtype=float)
    x = np.arange(arr.size, dtype=float)
    filled = arr.copy()
    filled[~mask] = np.interp(x[~mask], x[mask], arr[mask])
    return filled



def _resample(values: np.ndarray, time: np.ndarray, target_length: int) -> tuple[np.ndarray, np.ndarray]:
    if target_length < 8:
        raise ValueError("target_length must be at least 8")
    x = _fill_nan(values)
    t = np.asarray(time, dtype=float).reshape(-1)
    if x.size != t.size:
        raise ValueError("values and time must have the same length")
    if x.size == target_length:
        return x.astype(float), t.astype(float)
    t0 = float(t[0])
    t1 = float(t[-1])
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        src = np.linspace(0.0, 1.0, num=x.size, dtype=float)
        dst = np.linspace(0.0, 1.0, num=target_length, dtype=float)
    else:
        src = (t - t0) / (t1 - t0)
        dst = np.linspace(0.0, 1.0, num=target_length, dtype=float)
    y = np.interp(dst, src, x)
    return y.astype(float), dst.astype(float)



def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sigma



def _apply_transform(values: np.ndarray, *, transform: str = "zscore", difference: bool = False, log_return: bool = False) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    if log_return:
        safe = np.where(np.abs(x) < 1e-12, np.nan, x)
        x = np.diff(np.log(np.abs(safe)), prepend=np.log(np.abs(safe[0])))
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    elif difference:
        x = np.diff(x, prepend=x[0])
    transform_norm = (transform or "zscore").lower()
    if transform_norm in {"zscore", "z", "standard"}:
        return _zscore(x)
    if transform_norm in {"minmax", "range"}:
        lo = float(np.min(x))
        hi = float(np.max(x))
        span = hi - lo
        if not np.isfinite(span) or span <= 1e-12:
            return np.zeros_like(x)
        return (x - lo) / span
    if transform_norm in {"center", "demean"}:
        return x - float(np.mean(x))
    if transform_norm in {"none", "identity", "raw"}:
        return x.astype(float)
    raise ValueError(f"Unsupported transform: {transform}")



def _bounded_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 3:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(0.5 * (corr + 1.0), 0.0, 1.0))



def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip((float(np.dot(a, b)) / denom + 1.0) / 2.0, 0.0, 1.0))



def _spectral_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 8 or b.size < 8:
        return 0.0
    fa, pa = periodogram(a, scaling="density")
    fb, pb = periodogram(b, scaling="density")
    if len(fa) < 3 or len(fb) < 3:
        return 0.0
    fa = fa[1:]
    fb = fb[1:]
    pa = pa[1:]
    pb = pb[1:]
    grid = np.linspace(0.0, float(min(fa.max(), fb.max())), num=min(len(fa), len(fb)), dtype=float)
    if grid.size < 3 or grid[-1] <= 0:
        return 0.0
    pa_i = np.interp(grid, fa, pa)
    pb_i = np.interp(grid, fb, pb)
    pa_i = pa_i / max(float(np.sum(pa_i)), 1e-12)
    pb_i = pb_i / max(float(np.sum(pb_i)), 1e-12)
    return _cosine_similarity(pa_i, pb_i)



def _dtw_similarity(a: np.ndarray, b: np.ndarray, *, window_ratio: float = 0.1) -> float:
    n = int(a.size)
    m = int(b.size)
    if n == 0 or m == 0:
        return 0.0
    window = max(abs(n - m), int(max(n, m) * window_ratio))
    inf = float("inf")
    cost = np.full((n + 1, m + 1), inf, dtype=float)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        ai = a[i - 1]
        for j in range(j_start, j_end + 1):
            dist = abs(ai - b[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    dist = float(cost[n, m])
    if not np.isfinite(dist):
        return 0.0
    norm = dist / max(float(n + m), 1.0)
    return float(np.clip(exp(-norm), 0.0, 1.0))



def _turning_point_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 4 or b.size < 4:
        return 0.0
    da = np.sign(np.diff(a))
    db = np.sign(np.diff(b))
    agree = float(np.mean(da == db))
    return float(np.clip(agree, 0.0, 1.0))



def _volatility_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return 0.0
    va = float(np.std(np.diff(a)))
    vb = float(np.std(np.diff(b)))
    denom = max(abs(va) + abs(vb), 1e-12)
    return float(np.clip(1.0 - abs(va - vb) / denom, 0.0, 1.0))



def _tag_overlap(a_raw: np.ndarray, a_time: np.ndarray, b_raw: np.ndarray, b_time: np.ndarray) -> tuple[list[str], dict[str, Any]]:
    desc_a = describe_series(a_raw, a_time)
    desc_b = describe_series(b_raw, b_time)
    overlap = sorted(set(desc_a.inferred_tags) & set(desc_b.inferred_tags))
    evidence = {
        "reference_tags": list(desc_a.inferred_tags),
        "candidate_tags": list(desc_b.inferred_tags),
        "reference_scores": dict(desc_a.scores),
        "candidate_scores": dict(desc_b.scores),
    }
    return overlap, evidence



def _build_summary(name_a: str, name_b: str, aggregate: float, metrics: dict[str, float], tag_overlap: list[str]) -> str:
    strongest = sorted(metrics.items(), key=lambda item: item[1], reverse=True)
    top_metrics = ", ".join([f"{k}={v:.2f}" for k, v in strongest[:3]])
    if aggregate >= 0.8:
        verdict = "very similar"
    elif aggregate >= 0.65:
        verdict = "meaningfully similar"
    elif aggregate >= 0.5:
        verdict = "partially similar"
    else:
        verdict = "not strongly similar"
    overlap_text = f" Shared tags: {', '.join(tag_overlap)}." if tag_overlap else ""
    return (
        f"`{name_a}` and `{name_b}` look {verdict} after alignment and normalization. "
        f"The strongest signals come from {top_metrics}." + overlap_text
    )


DEFAULT_WEIGHTS: dict[str, float] = {
    "correlation": 0.35,
    "dtw": 0.35,
    "spectral": 0.20,
    "turning_points": 0.10,
    "volatility": 0.10,
}


def compare_series(
    reference: ArrayLike,
    candidate: ArrayLike,
    *,
    reference_time: ArrayLike | None = None,
    candidate_time: ArrayLike | None = None,
    reference_name: str | None = None,
    candidate_name: str | None = None,
    target_length: int = 256,
    transform: str = "zscore",
    difference: bool = False,
    log_return: bool = False,
    metrics: tuple[str, ...] = ("correlation", "dtw", "spectral", "turning_points"),
    weights: Mapping[str, float] | None = None,
) -> SimilarityResult:
    """Compare two time series with shape, alignment, and spectral metrics.

    Parameters
    ----------
    reference, candidate:
        Arrays or ``GeneratedSeries`` objects. If multivariate, the first
        channel is used.
    reference_time, candidate_time:
        Optional time bases. If omitted for ``GeneratedSeries``, the embedded
        time is used.
    transform:
        Pre-alignment normalization: ``"zscore"`` (default), ``"minmax"``,
        ``"center"``, or ``"none"``.
    difference:
        Compare first differences instead of levels. Useful for cumulative
        counts such as GitHub stars.
    log_return:
        Compare approximate log returns. Useful for prices and other positive
        level series.
    """

    a_raw, a_time, auto_a = _coerce_series(reference, reference_time)
    b_raw, b_time, auto_b = _coerce_series(candidate, candidate_time)
    a_name = reference_name or auto_a
    b_name = candidate_name or auto_b

    a_res, a_t = _resample(a_raw, a_time, target_length)
    b_res, b_t = _resample(b_raw, b_time, target_length)
    a = _apply_transform(a_res, transform=transform, difference=difference, log_return=log_return)
    b = _apply_transform(b_res, transform=transform, difference=difference, log_return=log_return)

    requested = tuple(metrics)
    metric_scores: dict[str, float] = {}
    for name in requested:
        key = name.lower()
        if key == "correlation":
            metric_scores[key] = _bounded_corr(a, b)
        elif key == "dtw":
            metric_scores[key] = _dtw_similarity(a, b)
        elif key == "spectral":
            metric_scores[key] = _spectral_similarity(a, b)
        elif key in {"turning_points", "turning", "direction"}:
            metric_scores["turning_points"] = _turning_point_similarity(a, b)
        elif key == "volatility":
            metric_scores[key] = _volatility_similarity(a, b)
        else:
            raise ValueError(f"Unsupported similarity metric: {name}")

    weight_map = {**DEFAULT_WEIGHTS, **{str(k): float(v) for k, v in (weights or {}).items()}}
    num = 0.0
    den = 0.0
    for key, value in metric_scores.items():
        w = float(weight_map.get(key, 1.0))
        num += w * float(value)
        den += w
    aggregate = float(num / den) if den > 0 else 0.0

    overlap, tag_evidence = _tag_overlap(a_raw, a_time, b_raw, b_time)
    evidence: dict[str, Any] = {
        "reference_range": [float(np.nanmin(a_raw)), float(np.nanmax(a_raw))],
        "candidate_range": [float(np.nanmin(b_raw)), float(np.nanmax(b_raw))],
        "difference": bool(difference),
        "log_return": bool(log_return),
        "tag_evidence": tag_evidence,
        "aligned_time_reference": a_t.tolist(),
        "aligned_time_candidate": b_t.tolist(),
    }
    summary = _build_summary(a_name, b_name, aggregate, metric_scores, overlap)
    return SimilarityResult(
        reference_name=a_name,
        candidate_name=b_name,
        aggregate_score=aggregate,
        metrics=metric_scores,
        transform=transform,
        target_length=target_length,
        tag_overlap=overlap,
        evidence=evidence,
        summary=summary,
    )



def pairwise_similarity(
    series_map: Mapping[str, ArrayLike],
    *,
    time_map: Mapping[str, ArrayLike] | None = None,
    target_length: int = 256,
    transform: str = "zscore",
    difference: bool = False,
    log_return: bool = False,
    metrics: tuple[str, ...] = ("correlation", "dtw", "spectral", "turning_points"),
    weights: Mapping[str, float] | None = None,
) -> SimilarityMatrix:
    """Compute a pairwise similarity matrix for named series."""

    names = list(series_map.keys())
    n = len(names)
    arr = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            ti = None if time_map is None else time_map.get(names[i])
            tj = None if time_map is None else time_map.get(names[j])
            res = compare_series(
                series_map[names[i]],
                series_map[names[j]],
                reference_time=ti,
                candidate_time=tj,
                reference_name=names[i],
                candidate_name=names[j],
                target_length=target_length,
                transform=transform,
                difference=difference,
                log_return=log_return,
                metrics=metrics,
                weights=weights,
            )
            arr[i, j] = res.aggregate_score
            arr[j, i] = res.aggregate_score
    return SimilarityMatrix(names=names, matrix=arr, metric="aggregate", transform=transform)



def find_top_matches(
    reference: ArrayLike,
    candidates: Mapping[str, ArrayLike],
    *,
    reference_time: ArrayLike | None = None,
    candidate_times: Mapping[str, ArrayLike] | None = None,
    reference_name: str = "reference",
    top_k: int = 5,
    target_length: int = 256,
    transform: str = "zscore",
    difference: bool = False,
    log_return: bool = False,
    metrics: tuple[str, ...] = ("correlation", "dtw", "spectral", "turning_points"),
    weights: Mapping[str, float] | None = None,
) -> list[SimilarityResult]:
    """Rank candidate series by similarity to a reference series."""

    results: list[SimilarityResult] = []
    for name, candidate in candidates.items():
        candidate_time = None if candidate_times is None else candidate_times.get(name)
        results.append(
            compare_series(
                reference,
                candidate,
                reference_time=reference_time,
                candidate_time=candidate_time,
                reference_name=reference_name,
                candidate_name=name,
                target_length=target_length,
                transform=transform,
                difference=difference,
                log_return=log_return,
                metrics=metrics,
                weights=weights,
            )
        )
    results.sort(key=lambda item: item.aggregate_score, reverse=True)
    return results[: max(1, int(top_k))]



def explain_similarity(result: SimilarityResult) -> str:
    """Return the human-readable explanation stored in a similarity result."""

    return result.summary
