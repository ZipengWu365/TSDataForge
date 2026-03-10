from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core.rng import as_rng
from .series import GeneratedSeries
from .specs import SeriesSpec
from .trace import SeriesTrace


@dataclass
class Compiler:
    seed: int | None = None

    def compile(self, spec: SeriesSpec, length: int) -> GeneratedSeries:
        rng = as_rng(self.seed)
        time = spec.observation.sampling.generate_time(length, rng)
        latent_result = spec.latent.evaluate(time, rng, context={"length": length}, path="latent")
        latent = np.asarray(latent_result.values, dtype=float)
        observed_time = time.copy()
        observed = latent.copy()
        states = dict(latent_result.states)
        masks: dict[str, np.ndarray] = {}

        if spec.observation.measurement_noise is not None:
            observed = spec.observation.measurement_noise.apply(observed, rng)
            states["observation/measurement_noise_std"] = float(spec.observation.measurement_noise.std)

        if spec.observation.missing is not None:
            observed_mask = spec.observation.missing.generate_mask(len(observed), rng)
            masks["observed_mask"] = observed_mask
            observed = observed.copy()
            observed[~observed_mask] = np.nan

        for idx, transform in enumerate(spec.observation.transforms):
            observed_time, observed, meta = transform.apply(observed_time, observed, rng)
            states[f"observation/transform_{idx}"] = meta

        tag_set = set(spec.tags)
        tag_set.update(latent_result.tags)
        trace = SeriesTrace(
            seed=self.seed,
            spec=spec.to_dict(),
            structure_id=spec.structure_id,
            time=time,
            observed_time=observed_time,
            latent=latent,
            observed=observed,
            contributions=latent_result.contributions,
            states=states,
            masks=masks,
            tags=tuple(sorted(tag_set)),
        )
        return GeneratedSeries(time=observed_time, values=observed, spec=spec, trace=trace)
