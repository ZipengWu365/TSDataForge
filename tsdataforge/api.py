from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

from .compiler import Compiler
from .datasets.builder import generate_dataset as build_dataset
from .specs import ObservationSpec, SeriesSpec
from .operators import Add
from .series import GeneratedSeries


def _build_spec_from_components(
    components: Sequence[object],
    observation: ObservationSpec | None = None,
    structure_id: str = "custom",
    tags: Sequence[str] = (),
) -> SeriesSpec:
    if not components:
        raise ValueError("At least one component is required.")
    latent = components[0] if len(components) == 1 else Add(tuple(components))
    return SeriesSpec(
        latent=latent,
        observation=observation or ObservationSpec(),
        structure_id=structure_id,
        tags=tuple(tags),
        name=structure_id,
    )


def generate_series(
    *,
    length: int,
    spec: SeriesSpec | None = None,
    components: Sequence[object] | None = None,
    observation: ObservationSpec | None = None,
    seed: int | None = None,
    return_trace: bool = True,
    structure_id: str = "custom",
    tags: Sequence[str] = (),
) -> GeneratedSeries:
    """Compile a time-series spec into a concrete series sample.

    Parameters
    ----------
    length:
        Number of time points to generate before observation transforms.
    spec:
        Explicit series spec. If omitted, ``components`` are wrapped into an additive spec.
    components:
        Components used to create a simple additive latent spec.
    observation:
        Optional observation spec used only when ``spec`` is omitted.
    seed:
        Seed for deterministic generation.
    return_trace:
        If ``False``, the returned series keeps values and time but drops the rich trace payload.
    """
    if spec is None:
        if components is None:
            raise ValueError("Provide either `spec` or `components`.")
        spec = _build_spec_from_components(components, observation=observation, structure_id=structure_id, tags=tags)
    elif components is not None:
        raise ValueError("Use either `spec` or `components`, not both.")
    elif observation is not None:
        spec = replace(spec, observation=observation)

    sample = Compiler(seed=seed).compile(spec, length=length)
    if return_trace:
        return sample
    return GeneratedSeries(time=sample.time, values=sample.values, spec=sample.spec, trace=None)


def compose_series(
    components: Sequence[object],
    *,
    length: int,
    seed: int | None = None,
    observation: ObservationSpec | None = None,
    return_trace: bool = True,
    structure_id: str = "custom",
    tags: Sequence[str] = (),
) -> GeneratedSeries:
    return generate_series(
        length=length,
        components=components,
        observation=observation,
        seed=seed,
        return_trace=return_trace,
        structure_id=structure_id,
        tags=tags,
    )


def generate_dataset(*args, **kwargs):
    return build_dataset(*args, **kwargs)
