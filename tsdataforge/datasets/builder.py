from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..specs import ObservationSpec, SeriesSpec
from .series_dataset import generate_series_dataset


@dataclass
class TaskDataset:
    task: str
    X: np.ndarray
    y: np.ndarray | None = None
    time: np.ndarray | None = None
    masks: dict[str, np.ndarray] | None = None
    aux: dict[str, np.ndarray] | None = None
    meta: list[dict[str, Any]] | None = None
    label_names: list[str] | None = None
    schema: dict[str, Any] | None = None

    def agent_context(self, **kwargs):
        from ..agent import build_task_context

        return build_task_context(self, **kwargs)

    def handoff(self, **kwargs):
        """Build a handoff bundle for this task dataset."""

        from ..agent import build_dataset_handoff_bundle

        return build_dataset_handoff_bundle(self, **kwargs)

    def save(self, directory: str | Path, *, include_card: bool = True, include_agent_context: bool = True, include_handoff_bundle: bool = True) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {"X": self.X}
        if self.y is not None:
            arrays["y"] = self.y
        if self.time is not None:
            arrays["time"] = self.time
        for key, value in (self.masks or {}).items():
            arrays[f"mask__{key}"] = value
        for key, value in (self.aux or {}).items():
            arrays[f"aux__{key}"] = value
        np.savez(directory / "dataset.npz", **arrays)
        manifest = {
            "task": self.task,
            "label_names": self.label_names,
            "schema": self.schema,
            "n_samples": int(len(self.X)),
            "meta": self.meta or [],
        }
        with (directory / "manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        if include_agent_context:
            from ..agent import build_task_context

            pack = build_task_context(self, budget="small")
            (directory / "task_context.json").write_text(json.dumps(pack.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            (directory / "task_context.md").write_text(pack.to_markdown(), encoding="utf-8")

        if include_card:
            from ..agent import build_task_dataset_card

            card = build_task_dataset_card(self)
            (directory / "task_card.json").write_text(json.dumps(card.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            (directory / "task_card.md").write_text(card.to_markdown(), encoding="utf-8")
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



def generate_dataset(
    *,
    task: str,
    structures: Sequence[str | SeriesSpec] | None = None,
    n_series: int = 100,
    length: int = 256,
    horizon: int = 32,
    seed: int | None = None,
    mask_ratio: float = 0.15,
    order_segments: int = 4,
    anomaly_rate: float = 0.02,
    observation_factory: Callable[[np.random.Generator], ObservationSpec] | None = None,
    include_aux: bool = False,
    window: int | None = None,
    stride: int = 1,
    outcome_channel: int | None = None,
    gamma: float = 1.0,
):
    """Generate a task-specialized dataset.

    v0.2.5 routes all generation through the base ``SeriesDataset`` +
    ``taskify_dataset`` pipeline so the task semantics stay aligned between
    synthetic and real/external datasets.
    """

    from .taskify import taskify_dataset

    base = generate_series_dataset(
        structures=structures,
        n_series=n_series,
        length=length,
        seed=seed,
        observation_factory=observation_factory,
        sampling="balanced" if str(task) == "classification" else "random",
        return_trace=True,
    )
    return taskify_dataset(
        base,
        task=task,
        horizon=horizon,
        mask_ratio=mask_ratio,
        order_segments=order_segments,
        anomaly_rate=anomaly_rate,
        seed=seed,
        window=window,
        stride=stride,
        outcome_channel=outcome_channel,
        include_aux=include_aux,
        gamma=gamma,
    )
