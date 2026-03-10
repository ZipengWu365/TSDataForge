from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core.base import Serializable
from .core.registry import register_type


class Sampling(Serializable, ABC):
    @abstractmethod
    def generate_time(self, length: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class MissingMechanism(Serializable, ABC):
    @abstractmethod
    def generate_mask(self, length: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class Transform(Serializable, ABC):
    @abstractmethod
    def apply(
        self,
        time: np.ndarray,
        values: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        raise NotImplementedError


@register_type
@dataclass(frozen=True)
class RegularSampling(Sampling):
    dt: float = 1.0

    def generate_time(self, length: int, rng: np.random.Generator) -> np.ndarray:
        del rng
        return np.arange(length, dtype=float) * float(self.dt)


@register_type
@dataclass(frozen=True)
class IrregularSampling(Sampling):
    dt: float = 1.0
    jitter: float = 0.15
    min_step: float = 1e-3

    def generate_time(self, length: int, rng: np.random.Generator) -> np.ndarray:
        if length <= 0:
            raise ValueError("length must be positive")
        if length == 1:
            return np.array([0.0], dtype=float)
        intervals = rng.normal(loc=self.dt, scale=abs(self.jitter * self.dt), size=length - 1)
        intervals = np.clip(intervals, self.min_step, None)
        return np.concatenate([[0.0], np.cumsum(intervals)])


@register_type
@dataclass(frozen=True)
class BlockMissing(MissingMechanism):
    rate: float = 0.1
    block_min: int = 5
    block_max: int = 20

    def generate_mask(self, length: int, rng: np.random.Generator) -> np.ndarray:
        mask = np.ones(length, dtype=bool)
        target_missing = int(round(length * np.clip(self.rate, 0.0, 1.0)))
        missing = 0
        while missing < target_missing:
            start = int(rng.integers(0, max(length, 1)))
            width = int(rng.integers(self.block_min, self.block_max + 1))
            end = min(length, start + width)
            mask[start:end] = False
            missing = int(length - mask.sum())
            if mask.sum() == 0:
                break
        return mask


@register_type
@dataclass(frozen=True)
class MeasurementNoise(Serializable):
    std: float | tuple[float, ...] = 0.05

    def apply(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        std = np.asarray(self.std, dtype=float)
        if std.ndim == 0:
            noise = rng.normal(loc=0.0, scale=float(std), size=arr.shape)
        else:
            # Per-channel std (last dimension)
            if arr.ndim == 1:
                if std.shape != (1,):
                    raise ValueError("Per-channel std for 1D series must have length 1")
                noise = rng.normal(loc=0.0, scale=float(std[0]), size=arr.shape)
            elif arr.ndim == 2:
                if std.shape != (arr.shape[1],):
                    raise ValueError("Per-channel std must match the number of channels")
                noise = rng.normal(loc=0.0, scale=std, size=arr.shape)
            else:
                raise ValueError("MeasurementNoise only supports 1D or 2D arrays")
        return arr + noise


@register_type
@dataclass(frozen=True)
class Clamp(Transform):
    min_value: float = -np.inf
    max_value: float = np.inf

    def apply(
        self,
        time: np.ndarray,
        values: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        del rng
        clipped = np.clip(values, self.min_value, self.max_value)
        return time, clipped, {"type": "clamp", "min_value": self.min_value, "max_value": self.max_value}


@register_type
@dataclass(frozen=True)
class Downsample(Transform):
    factor: int = 2
    agg: str = "mean"

    def apply(
        self,
        time: np.ndarray,
        values: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        del rng
        if self.factor <= 1:
            return time, values, {"type": "downsample", "factor": self.factor, "agg": self.agg}
        values_arr = np.asarray(values, dtype=float)
        n = len(values_arr) // self.factor
        if n <= 0:
            raise ValueError("Downsample factor is larger than the series length.")
        trimmed_time = time[: n * self.factor].reshape(n, self.factor)
        if values_arr.ndim == 1:
            trimmed_values = values_arr[: n * self.factor].reshape(n, self.factor)
        elif values_arr.ndim == 2:
            d = values_arr.shape[1]
            trimmed_values = values_arr[: n * self.factor].reshape(n, self.factor, d)
        else:
            raise ValueError("Downsample only supports 1D or 2D arrays")
        if self.agg == "mean":
            reduced_values = np.nanmean(trimmed_values, axis=1)
        elif self.agg == "median":
            reduced_values = np.nanmedian(trimmed_values, axis=1)
        else:
            raise ValueError(f"Unsupported agg={self.agg!r}")
        reduced_time = np.mean(trimmed_time, axis=1)
        return reduced_time, reduced_values, {"type": "downsample", "factor": self.factor, "agg": self.agg}
