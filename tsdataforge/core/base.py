from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from typing import Any, Mapping

import numpy as np

from .registry import serialize_value, type_key
from .results import EvalResult


class Serializable:
    """Dataclass mixin with a registry-backed dictionary representation."""

    def to_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} must be a dataclass to use Serializable.")
        return {
            "type": type_key(self.__class__),
            "params": {field.name: serialize_value(getattr(self, field.name)) for field in fields(self)},
        }


class Component(Serializable, ABC):
    canonical_tags: tuple[str, ...] = ()

    def label(self) -> str:
        return getattr(self, "name", None) or self.__class__.__name__.lower()

    def tags(self) -> tuple[str, ...]:
        return tuple(self.canonical_tags)

    def children(self) -> tuple["Component", ...]:
        return ()

    def evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, Any] | None = None,
        path: str = "latent",
    ) -> EvalResult:
        arr = np.asarray(t, dtype=float)
        ctx = dict(context or {})
        result = self._evaluate(arr, rng, ctx, path)
        result.tags.update(self.tags())
        if not result.contributions and not self.children():
            result.contributions[path] = result.values.copy()
        return result

    @abstractmethod
    def _evaluate(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        context: Mapping[str, Any],
        path: str,
    ) -> EvalResult:
        raise NotImplementedError
