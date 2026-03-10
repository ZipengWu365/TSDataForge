from __future__ import annotations

import numpy as np


def segment_shuffle(
    values: np.ndarray,
    rng: np.random.Generator,
    n_segments: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    segments = np.array_split(np.asarray(values, dtype=float), n_segments)
    order = rng.permutation(n_segments)
    shuffled = np.concatenate([segments[i] for i in order])
    return shuffled, order.astype(int)
