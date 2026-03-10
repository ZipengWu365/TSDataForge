from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.signal import periodogram

from ..operators import Add, Stack
from ..primitives.noise import AR1Noise, WhiteGaussianNoise
from ..primitives.seasonal import SineSeasonality
from ..primitives.trend import LinearTrend
from ..specs import SeriesSpec


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError("values must be 1D or 2D")


def _nan_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = float(np.sqrt(np.sum(aa**2) * np.sum(bb**2)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def _linear_fit(time: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(time) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan"), float("nan")
    t = time[mask]
    v = y[mask]
    t0 = float(np.mean(t))
    v0 = float(np.mean(v))
    denom = float(np.sum((t - t0) ** 2))
    if denom <= 0:
        return float("nan"), float("nan")
    slope = float(np.sum((t - t0) * (v - v0)) / denom)
    intercept = float(v0 - slope * t0)
    pred = slope * t + intercept
    ss_res = float(np.sum((v - pred) ** 2))
    ss_tot = float(np.sum((v - v0) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return slope, r2


def _acf1(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return float("nan")
    return _nan_corr(y[:-1], y[1:])


def _dominant_periods(time: np.ndarray, y: np.ndarray, top_k: int = 3) -> list[dict[str, float]]:
    y = np.asarray(y, dtype=float)
    if len(y) < 8:
        return []
    # Fill NaNs with mean for spectral estimate
    mu = float(np.nanmean(y)) if np.isfinite(np.nanmean(y)) else 0.0
    yy = np.nan_to_num(y, nan=mu)
    # Detrend
    slope, _ = _linear_fit(time, yy)
    if np.isfinite(slope):
        yy = yy - slope * (time - float(time[0]))
    yy = yy - float(np.mean(yy))
    dt = np.diff(time)
    dt_mean = float(np.mean(dt)) if len(dt) > 0 else 1.0
    fs = 1.0 / max(dt_mean, 1e-9)
    freqs, pxx = periodogram(yy, fs=fs, scaling="density")
    if len(freqs) < 3:
        return []
    # Exclude DC
    freqs = freqs[1:]
    pxx = pxx[1:]
    total = float(np.sum(pxx))
    if not np.isfinite(total) or total <= 0:
        return []
    # Pick local maxima by sorting power
    idx = np.argsort(pxx)[::-1]
    peaks: list[dict[str, float]] = []
    used: set[int] = set()
    for i in idx:
        if len(peaks) >= top_k:
            break
        if int(i) in used:
            continue
        f = float(freqs[int(i)])
        if f <= 0:
            continue
        period = 1.0 / f
        # Avoid trivially short/long periods
        if period < 2 * dt_mean or period > (time[-1] - time[0]) / 2:
            continue
        power = float(pxx[int(i)])
        peaks.append({"frequency": f, "period": period, "power": power, "power_ratio": power / total})
        # Mark a small neighborhood to avoid near-duplicates
        for j in range(max(0, int(i) - 2), min(len(freqs), int(i) + 3)):
            used.add(j)
    return peaks


@dataclass
class SeriesDescription:
    length: int
    n_channels: int
    missing_rate: float
    dt_mean: float
    dt_cv: float
    per_channel: list[dict[str, Any]] = field(default_factory=list)
    cross_correlation_mean_abs: float | None = None
    dominant_periods: list[dict[str, Any]] = field(default_factory=list)
    inferred_tags: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_series(
    values: np.ndarray,
    time: np.ndarray | None = None,
    *,
    top_k_periods: int = 3,
) -> SeriesDescription:
    """Compute a lightweight, dependency-minimal description of a real time series.

    The goal is not perfect statistical testing; it's a pragmatic, explainable
    descriptor that can (a) tag common structures and (b) seed synthetic specs.
    """

    arr = _as_2d(values)
    n = int(arr.shape[0])
    d = int(arr.shape[1])
    if time is None:
        time = np.arange(n, dtype=float)
    else:
        time = np.asarray(time, dtype=float)
        if len(time) != n:
            raise ValueError("time length must match values length")

    missing_rate = float(np.mean(~np.isfinite(arr)))
    dt = np.diff(time)
    dt_mean = float(np.mean(dt)) if len(dt) else 1.0
    dt_std = float(np.std(dt)) if len(dt) else 0.0
    dt_cv = float(dt_std / dt_mean) if dt_mean > 0 else float("inf")

    per_channel: list[dict[str, Any]] = []
    slopes: list[float] = []
    r2s: list[float] = []
    acf1s: list[float] = []
    kurt: list[float] = []
    spiky: list[float] = []
    for j in range(d):
        y = arr[:, j]
        mu = float(np.nanmean(y)) if np.isfinite(np.nanmean(y)) else float("nan")
        sd = float(np.nanstd(y)) if np.isfinite(np.nanstd(y)) else float("nan")
        if not np.isfinite(sd) or sd == 0:
            sd = float("nan")
        # Moments
        yy = y[np.isfinite(y)]
        if len(yy) >= 3:
            m = float(np.mean(yy))
            s = float(np.std(yy))
            if s > 0:
                skew = float(np.mean(((yy - m) / s) ** 3))
                kurtosis = float(np.mean(((yy - m) / s) ** 4))
            else:
                skew = float("nan")
                kurtosis = float("nan")
        else:
            skew = float("nan")
            kurtosis = float("nan")

        slope, r2 = _linear_fit(time, y)
        a1 = _acf1(y)
        slopes.append(slope)
        r2s.append(r2)
        acf1s.append(a1)
        kurt.append(kurtosis)
        if len(yy) >= 10 and np.isfinite(np.nanstd(y)) and float(np.nanstd(y)) > 0:
            thr = 3.0 * float(np.nanstd(y))
            sp = float(np.mean(np.abs(np.nan_to_num(y, nan=0.0)) > thr))
        else:
            sp = float("nan")
        spiky.append(sp)

        per_channel.append(
            {
                "mean": mu,
                "std": float(np.nanstd(y)),
                "skew": skew,
                "kurtosis": kurtosis,
                "trend_slope": slope,
                "trend_r2": r2,
                "acf1": a1,
                "spikiness": sp,
            }
        )

    # Cross-correlation as a crude coupling indicator
    cross_corr = None
    if d > 1:
        corr_vals: list[float] = []
        for i in range(d):
            for j in range(i + 1, d):
                corr_vals.append(abs(_nan_corr(arr[:, i], arr[:, j])))
        if corr_vals:
            cross_corr = float(np.nanmean(corr_vals))

    # Dominant periods on the first channel (cheap default)
    dom_periods = _dominant_periods(time, arr[:, 0], top_k=top_k_periods)

    desc = SeriesDescription(
        length=n,
        n_channels=d,
        missing_rate=missing_rate,
        dt_mean=dt_mean,
        dt_cv=dt_cv,
        per_channel=per_channel,
        cross_correlation_mean_abs=cross_corr,
        dominant_periods=dom_periods,
    )
    tags, scores = infer_structure_tags(desc)
    desc.inferred_tags = tags
    desc.scores = scores
    return desc


def infer_structure_tags(desc: SeriesDescription) -> tuple[list[str], dict[str, float]]:
    """Heuristic taxonomy tagging for real-world time series.

    This is intentionally lightweight and explainable.
    """

    tags: list[str] = []
    scores: dict[str, float] = {}

    # Sampling/observation tags
    if desc.missing_rate > 0:
        tags.append("missing")
        scores["missing_rate"] = float(desc.missing_rate)
    if desc.dt_cv > 0.05:
        tags.append("irregular_sampling")
        scores["dt_cv"] = float(desc.dt_cv)

    if desc.n_channels > 1:
        tags.append("multivariate")
        if desc.cross_correlation_mean_abs is not None:
            scores["cross_corr_mean_abs"] = float(desc.cross_correlation_mean_abs)
            if desc.cross_correlation_mean_abs >= 0.5:
                tags.append("coupled")

    # Aggregate channel metrics
    slopes = np.array([c.get("trend_slope", np.nan) for c in desc.per_channel], dtype=float)
    r2s = np.array([c.get("trend_r2", np.nan) for c in desc.per_channel], dtype=float)
    acf1s = np.array([c.get("acf1", np.nan) for c in desc.per_channel], dtype=float)
    kurt = np.array([c.get("kurtosis", np.nan) for c in desc.per_channel], dtype=float)
    spiky = np.array([c.get("spikiness", np.nan) for c in desc.per_channel], dtype=float)

    slope_mag = float(np.nanmean(np.abs(slopes))) if slopes.size else float("nan")
    r2_med = float(np.nanmedian(r2s)) if r2s.size else float("nan")
    acf1_med = float(np.nanmedian(acf1s)) if acf1s.size else float("nan")

    # Trend detection (scale-free heuristic)
    if np.isfinite(slope_mag) and np.isfinite(r2_med) and (r2_med >= 0.25):
        tags.append("trend")
        scores["trend_r2_median"] = r2_med
        scores["trend_slope_mean_abs"] = slope_mag

    # Seasonality detection using dominant period power ratios
    if desc.dominant_periods:
        best = max(desc.dominant_periods, key=lambda p: float(p.get("power_ratio", 0.0)))
        ratio = float(best.get("power_ratio", 0.0))
        scores["seasonality_power_ratio"] = ratio
        if ratio >= 0.20:
            tags.append("seasonal")
            tags.append("single_periodic")
            scores["dominant_period"] = float(best.get("period", float("nan")))

    # Random walk / AR(1) crude discrimination
    if np.isfinite(acf1_med):
        scores["acf1_median"] = acf1_med
        if acf1_med >= 0.95:
            tags.append("random_walk_like")
            tags.append("nonstationary")
        elif acf1_med >= 0.35:
            tags.append("colored_noise")
            tags.append("ar1_like")
            tags.append("stationary")
        else:
            tags.append("white_noise_like")

    # Bursty / heavy-tailed
    kurt_med = float(np.nanmedian(kurt)) if kurt.size else float("nan")
    spiky_med = float(np.nanmedian(spiky)) if spiky.size else float("nan")
    if np.isfinite(kurt_med):
        scores["kurtosis_median"] = kurt_med
    if np.isfinite(spiky_med):
        scores["spikiness_median"] = spiky_med
    if (np.isfinite(kurt_med) and kurt_med >= 6.0) or (np.isfinite(spiky_med) and spiky_med >= 0.01):
        tags.append("bursty")
        tags.append("heavy_tail")

    # De-duplicate while keeping order
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out, scores


def suggest_spec(
    desc: SeriesDescription,
    *,
    channel_mode: Literal["per_channel", "shared"] = "per_channel",
) -> SeriesSpec:
    """Suggest a simple executable `SeriesSpec` approximating the described real series.

    This is meant as a *seed* for synthetic benchmarking or augmentation, not a
    statistically optimal fit.
    """

    n = int(desc.length)
    d = int(desc.n_channels)

    # Estimate a dominant period if present
    period = None
    if desc.dominant_periods:
        best = max(desc.dominant_periods, key=lambda p: float(p.get("power_ratio", 0.0)))
        if float(best.get("power_ratio", 0.0)) >= 0.10:
            period = float(best.get("period", 0.0))
            if not np.isfinite(period) or period <= 0:
                period = None

    def _channel_components(ch: int) -> Any:
        chd = desc.per_channel[min(max(ch, 0), len(desc.per_channel) - 1)] if desc.per_channel else {}
        slope = float(chd.get("trend_slope", 0.0)) if np.isfinite(chd.get("trend_slope", 0.0)) else 0.0
        comps: list[Any] = []
        if "trend" in desc.inferred_tags and np.isfinite(slope):
            comps.append(LinearTrend(slope=slope, intercept=0.0))
        if period is not None and "seasonal" in desc.inferred_tags:
            comps.append(SineSeasonality(freq=period, amp=1.0))
        # AR(1) phi from acf1
        acf1 = float(chd.get("acf1", 0.0)) if np.isfinite(chd.get("acf1", 0.0)) else 0.0
        if "ar1_like" in desc.inferred_tags and 0 < acf1 < 0.999:
            comps.append(AR1Noise(phi=float(np.clip(acf1, 0.0, 0.98)), sigma=0.1))
        else:
            comps.append(WhiteGaussianNoise(std=0.1))
        return comps[0] if len(comps) == 1 else Add(tuple(comps))

    if d == 1:
        latent = _channel_components(0)
    else:
        if channel_mode == "shared":
            shared = _channel_components(0)
            latent = Stack(tuple(shared for _ in range(d)))
        else:
            latent = Stack(tuple(_channel_components(i) for i in range(d)))

    return SeriesSpec(latent=latent, structure_id="suggested_from_real", tags=tuple(desc.inferred_tags), name="suggested")
