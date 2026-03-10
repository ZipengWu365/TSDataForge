from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..compiler import Compiler
from ..core.rng import as_rng
from ..series import GeneratedSeries
from ..specs import ObservationSpec, SeriesSpec
from ..taxonomy.recipes import build_recipe, recipe_names


def _stack_or_object(items: list[np.ndarray]) -> np.ndarray:
    try:
        return np.stack(items)
    except ValueError:
        out = np.empty(len(items), dtype=object)
        for i, item in enumerate(items):
            out[i] = item
        return out



def _resolve_structure(structure: str | SeriesSpec, rng: np.random.Generator) -> SeriesSpec:
    if isinstance(structure, SeriesSpec):
        return structure
    if isinstance(structure, str):
        return build_recipe(structure, seed=rng)
    raise TypeError(f"Unsupported structure type: {type(structure)!r}")


@dataclass
class SeriesDataset:
    """A collection of raw time series samples (+ optional specs/traces).

    This is the *base dataset* abstraction. Task datasets (forecasting,
    classification, causal response, ...) should be derived via `taskify()`.
    """

    series: list[GeneratedSeries]
    meta: list[dict[str, Any]] = field(default_factory=list)
    dataset_id: str = "series_dataset"
    channel_names: list[str] | None = None

    @classmethod
    def from_arrays(
        cls,
        values: Any,
        time: Any | None = None,
        *,
        meta: list[dict[str, Any]] | dict[str, Any] | None = None,
        dataset_id: str = "external",
        channel_names: list[str] | None = None,
    ) -> "SeriesDataset":
        """Create a `SeriesDataset` from real arrays.

        This is designed for *taskification* and *EDA* of real datasets.
        It intentionally does not guess labels like changepoints/events.
        """

        from ..primitives.noise import WhiteGaussianNoise

        if isinstance(values, np.ndarray):
            arr = np.asarray(values)
            if arr.ndim == 1:
                series_values = [arr]
            elif arr.ndim == 2:
                if time is not None and isinstance(time, np.ndarray) and np.asarray(time).ndim == 1 and len(time) == arr.shape[0]:
                    series_values = [arr]
                else:
                    series_values = [arr[i, :] for i in range(arr.shape[0])]
            elif arr.ndim == 3:
                series_values = [arr[i, :, :] for i in range(arr.shape[0])]
            else:
                raise ValueError("values ndarray must be 1D, 2D, or 3D")
        elif isinstance(values, (list, tuple)):
            series_values = [np.asarray(v) for v in values]
        else:
            raise TypeError("values must be ndarray or list/tuple of arrays")

        n = len(series_values)
        if n == 0:
            raise ValueError("Empty dataset")

        if time is None:
            series_time = [np.arange(len(v), dtype=float) for v in series_values]
        elif isinstance(time, np.ndarray):
            t = np.asarray(time, dtype=float)
            if t.ndim == 1:
                series_time = [t for _ in range(n)]
            elif t.ndim == 2:
                if t.shape[0] != n:
                    raise ValueError("time[0] dimension must match number of series")
                series_time = [t[i, :] for i in range(n)]
            else:
                raise ValueError("time ndarray must be 1D or 2D")
        elif isinstance(time, (list, tuple)):
            if len(time) != n:
                raise ValueError("time list length must match number of series")
            series_time = [np.asarray(ti, dtype=float) if ti is not None else np.arange(len(v), dtype=float) for ti, v in zip(time, series_values)]
        else:
            raise TypeError("time must be None, ndarray, or list/tuple")

        if meta is None:
            meta_list = [{"structure_id": dataset_id, "tags": ["external"]} for _ in range(n)]
        elif isinstance(meta, dict):
            meta_list = [dict(meta) for _ in range(n)]
        elif isinstance(meta, list):
            if len(meta) != n:
                raise ValueError("meta list length must match number of series")
            meta_list = [dict(mi) for mi in meta]
        else:
            raise TypeError("meta must be None, dict, or list of dict")

        series: list[GeneratedSeries] = []
        for v, t_i, m_i in zip(series_values, series_time, meta_list):
            sid = str(m_i.get("structure_id", dataset_id))
            spec = SeriesSpec(latent=WhiteGaussianNoise(std=0.0), observation=ObservationSpec(), structure_id=sid, tags=("external",), name=sid)
            series.append(GeneratedSeries(time=np.asarray(t_i, dtype=float), values=np.asarray(v, dtype=float), spec=spec, trace=None))
        return cls(series=series, meta=meta_list, dataset_id=dataset_id, channel_names=channel_names)

    def __len__(self) -> int:
        return len(self.series)

    def values_list(self) -> list[np.ndarray]:
        return [np.asarray(s.values) for s in self.series]

    def time_list(self) -> list[np.ndarray]:
        return [np.asarray(s.time) for s in self.series]

    def values(self) -> np.ndarray:
        return _stack_or_object(self.values_list())

    def time(self) -> np.ndarray:
        return _stack_or_object(self.time_list())

    def describe(self, **kwargs):
        from ..analysis.dataset import describe_dataset

        return describe_dataset(self.values_list(), self.time_list(), **kwargs)

    def agent_context(self, **kwargs):
        from ..agent import build_dataset_context

        return build_dataset_context(self, **kwargs)

    def handoff(self, **kwargs):
        """Build the shortest report + handoff bundle for this dataset."""

        from ..agent import build_dataset_handoff_bundle

        return build_dataset_handoff_bundle(self, **kwargs)

    def taskify(self, task: str, **kwargs):
        """Convert this base dataset into a task-specific dataset.

        See `tsdataforge.datasets.taskify.taskify_dataset` for details.
        """

        from .taskify import taskify_dataset

        return taskify_dataset(self, task=task, **kwargs)

    def save(self, directory: str | Path, *, include_trace_arrays: bool = False, include_card: bool = True, include_agent_context: bool = True, include_handoff_bundle: bool = True) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        values = self.values_list()
        time = self.time_list()
        np.savez(directory / "series_dataset.npz", values=_stack_or_object(values), time=_stack_or_object(time))

        manifest: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "n_series": int(len(self.series)),
            "channel_names": self.channel_names,
            "meta": self.meta,
        }

        if include_trace_arrays:
            aux: dict[str, Any] = {}
            for i, s in enumerate(self.series):
                if s.trace is None:
                    continue
                aux[f"trace__{i}__latent"] = s.trace.latent
                aux[f"trace__{i}__observed"] = s.trace.observed
            if aux:
                np.savez(directory / "trace_arrays.npz", **aux)
                manifest["trace_arrays"] = "trace_arrays.npz"

        with (directory / "manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        if include_agent_context:
            from ..agent import build_dataset_context

            pack = build_dataset_context(self, budget="small")
            (directory / "dataset_context.json").write_text(json.dumps(pack.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            (directory / "dataset_context.md").write_text(pack.to_markdown(), encoding="utf-8")

        if include_card:
            from ..agent import build_series_dataset_card

            card = build_series_dataset_card(self)
            (directory / "dataset_card.json").write_text(json.dumps(card.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            (directory / "dataset_card.md").write_text(card.to_markdown(), encoding="utf-8")
            (directory / "README.md").write_text(card.to_markdown(), encoding="utf-8")

        if include_handoff_bundle:
            from ..agent import build_dataset_handoff_bundle

            build_dataset_handoff_bundle(
                self,
                output_dir=directory,
                include_report=False,
                include_docs_site=False,
                include_source_asset=False,
            )



def generate_series_dataset(
    *,
    structures: Sequence[str | SeriesSpec] | None = None,
    n_series: int = 100,
    length: int = 256,
    seed: int | None = None,
    observation_factory: Callable[[np.random.Generator], ObservationSpec] | None = None,
    sampling: str = "random",
    return_trace: bool = True,
) -> SeriesDataset:
    """Generate a *base* dataset (raw series) from structure recipes/specs.

    This function intentionally does *not* shape X/y for a particular ML task.
    Use `SeriesDataset.taskify()` to derive forecasting/classification/causal
    datasets.
    """

    rng = as_rng(seed)
    structures = list(structures) if structures is not None else list(recipe_names())
    if not structures:
        raise ValueError("At least one structure must be provided.")

    series: list[GeneratedSeries] = []
    meta: list[dict[str, Any]] = []

    for i in range(int(n_series)):
        if sampling == "balanced":
            base_structure = structures[i % len(structures)]
        else:
            base_structure = structures[int(rng.integers(0, len(structures)))]
        spec = _resolve_structure(base_structure, rng)
        if observation_factory is not None:
            spec = replace(spec, observation=observation_factory(rng))
        sample_seed = int(rng.integers(0, 2**32 - 1))
        s = Compiler(seed=sample_seed).compile(spec, length=int(length))
        if not return_trace:
            s = GeneratedSeries(time=s.time, values=s.values, spec=s.spec, trace=None)
        series.append(s)

        meta.append(
            {
                "structure_id": spec.structure_id or spec.name or "custom",
                "seed": sample_seed,
                "tags": [] if s.trace is None else list(s.trace.tags),
                "spec": spec.to_dict(),
                "trace": None if s.trace is None else s.trace.to_metadata(),
            }
        )

    dataset_id = f"synthetic_{len(structures)}x{n_series}"
    return SeriesDataset(series=series, meta=meta, dataset_id=dataset_id)
