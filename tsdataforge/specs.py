from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core.base import Serializable
from .core.registry import register_type


@register_type
@dataclass(frozen=True)
class ObservationSpec(Serializable):
    sampling: Any = None
    missing: Any | None = None
    measurement_noise: Any | None = None
    transforms: tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.sampling is None:
            from .observation import RegularSampling

            object.__setattr__(self, "sampling", RegularSampling())


@register_type
@dataclass(frozen=True)
class SeriesSpec(Serializable):
    latent: Any
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    structure_id: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def merged_tags(self) -> tuple[str, ...]:
        if hasattr(self.latent, "tags"):
            latent_tags = tuple(self.latent.tags())
        else:
            latent_tags = ()
        seen: dict[str, None] = {}
        for tag in (*self.tags, *latent_tags):
            seen.setdefault(tag, None)
        return tuple(seen)
