from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SeriesTrace:
    seed: int | None
    spec: dict[str, Any]
    structure_id: str | None
    time: np.ndarray
    observed_time: np.ndarray
    latent: np.ndarray
    observed: np.ndarray
    contributions: dict[str, np.ndarray] = field(default_factory=dict)
    states: dict[str, Any] = field(default_factory=dict)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_metadata(self, include_arrays: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "seed": self.seed,
            "structure_id": self.structure_id,
            "tags": list(self.tags),
            "time_length": int(len(self.time)),
            "observed_length": int(len(self.observed_time)),
            "contribution_keys": sorted(self.contributions),
            "state_keys": sorted(self.states),
            "mask_keys": sorted(self.masks),
            "spec": self.spec,
        }
        if include_arrays:
            data["time"] = self.time.tolist()
            data["observed_time"] = self.observed_time.tolist()
            data["latent"] = self.latent.tolist()
            data["observed"] = self.observed.tolist()
        return data
