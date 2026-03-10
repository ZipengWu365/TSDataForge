from __future__ import annotations

import numpy as np


def block_mask(
    length: int,
    rng: np.random.Generator,
    mask_ratio: float = 0.15,
    block_min: int = 5,
    block_max: int = 20,
) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    target = int(round(length * np.clip(mask_ratio, 0.0, 1.0)))
    while int(mask.sum()) < target:
        start = int(rng.integers(0, max(length, 1)))
        width = int(rng.integers(block_min, block_max + 1))
        end = min(length, start + width)
        mask[start:end] = True
        if mask.all():
            break
    return mask


def apply_mask(values: np.ndarray, mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[mask] = fill_value
    return out
