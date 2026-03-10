from __future__ import annotations

import numpy as np


def as_rng(seed: int | np.integer | np.random.Generator | None = None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))
