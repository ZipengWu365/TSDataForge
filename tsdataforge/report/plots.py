from __future__ import annotations

import base64
import io
from typing import Iterable

import numpy as np


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Matplotlib is required for report plots. Install with: pip install tsdataforge[viz]"
        ) from e


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError("values must be 1D or 2D")


def _png_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    return data


def plot_series_overview(
    values: np.ndarray,
    time: np.ndarray,
    *,
    channel_names: list[str] | None = None,
    max_channels: int = 8,
) -> str:
    plt = _import_matplotlib()
    arr = _as_2d(values)
    t = np.asarray(time, dtype=float)
    c = int(arr.shape[1])
    show = min(c, int(max_channels))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for j in range(show):
        name = None
        if channel_names is not None and j < len(channel_names):
            name = channel_names[j]
        label = name if name else f"ch{j}"
        ax.plot(t, arr[:, j], label=label, linewidth=1.0)
    ax.set_title("Time series overview")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    if show > 1:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    out = _png_base64(fig)
    plt.close(fig)
    return out


def plot_missingness(values: np.ndarray, time: np.ndarray) -> str:
    plt = _import_matplotlib()
    arr = _as_2d(values)
    t = np.asarray(time, dtype=float)
    mask = np.isfinite(arr)
    fig, ax = plt.subplots(figsize=(10, 1.8))
    # Show as an image: 1=observed, 0=missing
    img = mask.astype(float).T
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_title("Observed mask (rows=channels)")
    ax.set_xlabel("time index")
    ax.set_ylabel("channel")
    fig.tight_layout()
    out = _png_base64(fig)
    plt.close(fig)
    return out


def plot_histogram(values: np.ndarray, *, bins: int = 60) -> str:
    plt = _import_matplotlib()
    arr = _as_2d(values)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    # Flatten all finite values
    x = arr[np.isfinite(arr)].ravel()
    if x.size == 0:
        x = np.asarray([0.0])
    ax.hist(x, bins=int(bins))
    ax.set_title("Value distribution (all channels)")
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()
    out = _png_base64(fig)
    plt.close(fig)
    return out


def _safe_periodogram(y: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import periodogram

    yy = np.asarray(y, dtype=float)
    yy = yy[np.isfinite(yy)]
    if yy.size < 8:
        return np.asarray([]), np.asarray([])
    yy = yy - float(np.mean(yy))
    fs = 1.0 / max(float(dt), 1e-9)
    f, p = periodogram(yy, fs=fs, scaling="density")
    return f, p


def plot_power_spectrum(values: np.ndarray, time: np.ndarray) -> str:
    plt = _import_matplotlib()
    arr = _as_2d(values)
    t = np.asarray(time, dtype=float)
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    f, p = _safe_periodogram(arr[:, 0], dt)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    if f.size == 0:
        ax.text(0.5, 0.5, "Not enough data for spectrum", ha="center", va="center")
        ax.set_axis_off()
    else:
        # drop DC
        if f.size > 1:
            f = f[1:]
            p = p[1:]
        ax.plot(f, p, linewidth=1.0)
        ax.set_title("Power spectrum (channel 0)")
        ax.set_xlabel("frequency")
        ax.set_ylabel("power")
    fig.tight_layout()
    out = _png_base64(fig)
    plt.close(fig)
    return out


def _acf(y: np.ndarray, max_lag: int = 80) -> np.ndarray:
    yy = np.asarray(y, dtype=float)
    yy = yy[np.isfinite(yy)]
    n = yy.size
    if n < 4:
        return np.asarray([])
    yy = yy - float(np.mean(yy))
    denom = float(np.sum(yy**2))
    if denom <= 0:
        return np.asarray([])
    max_lag = int(min(max_lag, n - 1))
    out = np.empty(max_lag + 1, dtype=float)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        out[k] = float(np.sum(yy[:-k] * yy[k:]) / denom)
    return out


def plot_acf(values: np.ndarray, *, max_lag: int = 80) -> str:
    plt = _import_matplotlib()
    arr = _as_2d(values)
    acf = _acf(arr[:, 0], max_lag=max_lag)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    if acf.size == 0:
        ax.text(0.5, 0.5, "Not enough data for ACF", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.bar(np.arange(acf.size), acf)
        ax.set_title("Autocorrelation (channel 0)")
        ax.set_xlabel("lag")
        ax.set_ylabel("acf")
        ax.set_ylim(-1.0, 1.0)
    fig.tight_layout()
    out = _png_base64(fig)
    plt.close(fig)
    return out


def plot_sampling_intervals(time: np.ndarray, *, bins: int = 50) -> str:
    plt = _import_matplotlib()
    t = np.asarray(time, dtype=float)
    dt = np.diff(t)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    if dt.size == 0:
        ax.text(0.5, 0.5, "No intervals", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(dt[np.isfinite(dt)], bins=int(bins))
        ax.set_title("Sampling interval distribution")
        ax.set_xlabel("dt")
        ax.set_ylabel("count")
    fig.tight_layout()
    out = _png_base64(fig)
    plt.close(fig)
    return out


def plot_cross_correlation(values: np.ndarray) -> str:
    plt = _import_matplotlib()
    arr = _as_2d(values)
    c = int(arr.shape[1])
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    if c <= 1:
        ax.text(0.5, 0.5, "Univariate", ha="center", va="center")
        ax.set_axis_off()
    else:
        # correlation matrix with NaN-handling
        mat = np.full((c, c), np.nan, dtype=float)
        for i in range(c):
            for j in range(c):
                a = arr[:, i]
                b = arr[:, j]
                mask = np.isfinite(a) & np.isfinite(b)
                if int(mask.sum()) < 3:
                    continue
                aa = a[mask] - float(np.mean(a[mask]))
                bb = b[mask] - float(np.mean(b[mask]))
                denom = float(np.sqrt(np.sum(aa**2) * np.sum(bb**2)))
                if denom <= 0:
                    continue
                mat[i, j] = float(np.sum(aa * bb) / denom)
        im = ax.imshow(mat, vmin=-1.0, vmax=1.0, interpolation="nearest")
        ax.set_title("Cross-correlation heatmap")
        ax.set_xlabel("channel")
        ax.set_ylabel("channel")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = _png_base64(fig)
    plt.close(fig)
    return out


def plot_decomposition(values: np.ndarray, time: np.ndarray, *, period: float | None = None) -> str:
    """A lightweight, dependency-minimal decomposition plot.

    - trend is estimated via a moving average
    - if `period` is provided, a naive seasonal pattern is estimated by averaging
      values at the same phase (mod period)
    """

    plt = _import_matplotlib()
    arr = _as_2d(values)
    y = np.asarray(arr[:, 0], dtype=float)
    t = np.asarray(time, dtype=float)

    # Fill missing for plotting
    mu = float(np.nanmean(y)) if np.isfinite(np.nanmean(y)) else 0.0
    yy = np.nan_to_num(y, nan=mu)

    n = len(yy)
    if n < 8:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(t, yy)
        ax.set_title("Decomposition (insufficient length)")
        fig.tight_layout()
        out = _png_base64(fig)
        plt.close(fig)
        return out

    # Trend via moving average
    win = int(max(5, min(n // 10, 201)))
    if period is not None and np.isfinite(period) and period > 2:
        win = int(max(5, min(int(round(period)), 401)))
    kernel = np.ones(win, dtype=float) / float(win)
    trend = np.convolve(yy, kernel, mode="same")
    resid = yy - trend

    seasonal = None
    if period is not None and np.isfinite(period) and period > 2:
        p = int(max(2, round(period)))
        phase = (np.arange(n) % p).astype(int)
        pat = np.zeros(p, dtype=float)
        cnt = np.zeros(p, dtype=float)
        for i in range(n):
            pat[phase[i]] += resid[i]
            cnt[phase[i]] += 1.0
        cnt[cnt == 0] = 1.0
        pat = pat / cnt
        seasonal = pat[phase]

    fig, axes = plt.subplots(3 if seasonal is not None else 2, 1, figsize=(10, 6.0), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes[0].plot(t, yy, linewidth=1.0)
    axes[0].set_title("Observed (channel 0)")
    axes[1].plot(t, trend, linewidth=1.0)
    axes[1].set_title(f"Trend (moving average, window={win})")
    if seasonal is not None:
        axes[2].plot(t, seasonal, linewidth=1.0)
        axes[2].set_title(f"Seasonal (naive phase average, period≈{period:.3g})")
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    out = _png_base64(fig)
    plt.close(fig)
    return out


def plot_tag_frequency(tag_counts: dict[str, int], *, top_k: int = 20) -> str:
    plt = _import_matplotlib()
    items = list(tag_counts.items())[: int(top_k)]
    labels = [k for k, _ in items]
    counts = [int(v) for _, v in items]
    fig, ax = plt.subplots(figsize=(9.0, 0.35 * max(6, len(items))))
    if not items:
        ax.text(0.5, 0.5, "No tags", ha="center", va="center")
        ax.set_axis_off()
    else:
        y = np.arange(len(items))
        ax.barh(y, counts)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title("Tag frequency (top)")
        ax.set_xlabel("count")
    fig.tight_layout()
    out = _png_base64(fig)
    plt.close(fig)
    return out


def plot_feature_hist(x: Iterable[float], title: str, *, bins: int = 40) -> str:
    plt = _import_matplotlib()
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    if arr.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(arr, bins=int(bins))
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    fig.tight_layout()
    out = _png_base64(fig)
    plt.close(fig)
    return out


def plot_adjacency_matrix(adjacency: np.ndarray) -> str:
    plt = _import_matplotlib()
    adj = np.asarray(adjacency, dtype=float)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    if adj.ndim != 2:
        ax.text(0.5, 0.5, "Adjacency must be 2D", ha="center", va="center")
        ax.set_axis_off()
    else:
        im = ax.imshow(adj, interpolation="nearest")
        ax.set_title("Adjacency / causal graph")
        ax.set_xlabel("source")
        ax.set_ylabel("target")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = _png_base64(fig)
    plt.close(fig)
    return out



def plot_factual_vs_counterfactual(
    values: np.ndarray,
    counterfactual: np.ndarray,
    time: np.ndarray,
    *,
    outcome_channel: int | None = None,
    label: str = "counterfactual",
) -> str:
    plt = _import_matplotlib()
    arr = _as_2d(values)
    cf = _as_2d(counterfactual)
    t = np.asarray(time, dtype=float)
    ch = int(outcome_channel) if outcome_channel is not None else int(min(arr.shape[1], cf.shape[1]) - 1)
    ch = max(0, min(ch, arr.shape[1] - 1, cf.shape[1] - 1))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t, arr[:, ch], label="factual", linewidth=1.2)
    ax.plot(t, cf[:, ch], label=label, linewidth=1.2)
    ax.set_title(f"Factual vs {label} (channel {ch})")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = _png_base64(fig)
    plt.close(fig)
    return out



def plot_binary_mask(time: np.ndarray, mask: np.ndarray, *, title: str = "Binary mask") -> str:
    plt = _import_matplotlib()
    t = np.asarray(time, dtype=float)
    m = np.asarray(mask).astype(float).reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 1.8))
    if len(m) != len(t):
        ax.text(0.5, 0.5, "Mask length mismatch", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.step(t, m, where="post")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("active")
    fig.tight_layout()
    out = _png_base64(fig)
    plt.close(fig)
    return out
