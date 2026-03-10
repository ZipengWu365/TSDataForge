from __future__ import annotations

import numpy as np


def _jitter(values: np.ndarray, rng: np.random.Generator, scale: float = 0.03) -> np.ndarray:
    std = np.nanstd(values)
    std = 1.0 if not np.isfinite(std) or std == 0 else std
    arr = np.asarray(values, dtype=float)
    return arr + rng.normal(scale=scale * std, size=arr.shape)


def _scale(values: np.ndarray, rng: np.random.Generator, low: float = 0.8, high: float = 1.2) -> np.ndarray:
    return values * float(rng.uniform(low, high))


def _drift(values: np.ndarray, rng: np.random.Generator, strength: float = 0.05) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        drift = np.linspace(0.0, float(rng.normal(scale=strength)), num=len(arr))
        return arr + drift
    if arr.ndim == 2:
        t = np.linspace(0.0, 1.0, num=len(arr))[:, None]
        amp = rng.normal(scale=strength, size=(1, arr.shape[1]))
        return arr + t * amp
    raise ValueError("contrastive drift only supports 1D or 2D arrays")


def random_view(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    ops = (_jitter, _scale, _drift)
    chosen = rng.choice(len(ops), size=2, replace=False)
    for idx in chosen:
        out = ops[int(idx)](out, rng)
    return out


def make_contrastive_pair(values: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(values, dtype=float)
    return random_view(base, rng), random_view(base, rng)
