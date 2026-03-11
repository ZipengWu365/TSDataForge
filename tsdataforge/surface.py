from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .agent.context import build_dataset_context
from .agent.decision import build_dataset_decision_record
from .agent.handoff import DatasetHandoffBundle, build_dataset_handoff_bundle
from .datasets.builder import TaskDataset
from .datasets.series_dataset import SeriesDataset
from .datasets.taskify import taskify_dataset
from .loading import coerce_asset
from .report.eda import EDAReport, generate_dataset_eda_report, generate_eda_report
from .series import GeneratedSeries


@dataclass(frozen=True)
class PublicEntrypoint:
    name: str
    signature: str
    one_liner: str
    returns: str
    why_exists: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PublicSurface:
    version: str
    entrypoints: tuple[PublicEntrypoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "entrypoints": [item.to_dict() for item in self.entrypoints]}

    def to_markdown(self) -> str:
        lines = [
            f"# TSDataForge public surface v{self.version}",
            "",
            "These are the five entry points new users should memorize first.",
            "",
            "| API | What it does | Returns | Why it exists |",
            "|---|---|---|---|",
        ]
        for item in self.entrypoints:
            lines.append(f"| `{item.signature}` | {item.one_liner} | `{item.returns}` | {item.why_exists} |")
        return "\n".join(lines).strip() + "\n"


def public_surface(version: str = "0.3.7") -> PublicSurface:
    return PublicSurface(
        version=version,
        entrypoints=(
            PublicEntrypoint(
                "load_asset",
                "load_asset(source, time=None, dataset_id=None)",
                "Turn a path or raw arrays into a reusable TSDataForge asset.",
                "SeriesDataset | TaskDataset",
                "New users should not have to guess how to coerce `.npy`, `.npz`, `.csv`, or raw arrays.",
            ),
            PublicEntrypoint(
                "report",
                'report(source, output_path="report.html")',
                "Generate the first human-readable EDA artifact from a series or dataset.",
                "EDAReport",
                "The package should feel like a dataset report layer before it feels like a toolkit.",
            ),
            PublicEntrypoint(
                "handoff",
                'handoff(source, output_dir="handoff_bundle")',
                "Package report, context, card, and next actions into one predictable bundle.",
                "DatasetHandoffBundle",
                "This is the shortest happy path from raw asset to reusable, shareable output.",
            ),
            PublicEntrypoint(
                "taskify",
                "taskify(source, task=..., ...)",
                "Derive a task-specific dataset only after the asset and report are clear.",
                "TaskDataset",
                "Taskification should come after understanding, not before it.",
            ),
            PublicEntrypoint(
                "demo",
                "demo(output_dir=\"demo_bundle\", scenario=...)",
                "Generate a built-in demo asset and the full public handoff flow.",
                "DatasetHandoffBundle",
                "A credible project needs a copy-paste first success for GitHub, workshops, and agents.",
            ),
        ),
    )


def render_public_surface_markdown(surface: PublicSurface | None = None) -> str:
    return (surface or public_surface()).to_markdown()


def save_public_surface(surface: PublicSurface | None, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = surface or public_surface()
    if out.suffix.lower() == ".json":
        out.write_text(json.dumps(payload.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return
    out.write_text(payload.to_markdown(), encoding="utf-8")


def load_asset(
    source: Any,
    time: Any | None = None,
    *,
    dataset_id: str | None = None,
    channel_names: list[str] | None = None,
) -> SeriesDataset | TaskDataset:
    return coerce_asset(source, time, dataset_id=dataset_id, channel_names=channel_names)


def report(
    source: Any,
    time: Any | None = None,
    *,
    output_path: str | Path | None = "report.html",
    title: str | None = None,
    docs_base_url: str | None = None,
    dataset_id: str | None = None,
    channel_names: list[str] | None = None,
) -> EDAReport:
    asset = (
        source
        if isinstance(source, (SeriesDataset, TaskDataset, GeneratedSeries))
        else coerce_asset(source, time, dataset_id=dataset_id, channel_names=channel_names)
    )
    if isinstance(asset, TaskDataset):
        raise TypeError("`report()` expects a series or base dataset. Use `task.handoff(...)` or `handoff(task_dataset, ...)` for task assets.")
    if isinstance(asset, SeriesDataset):
        decision = build_dataset_decision_record(build_dataset_context(asset, budget="small"))
        return generate_dataset_eda_report(
            asset.values_list(),
            asset.time_list(),
            title=title or f"TSDataForge Dataset Report - {asset.dataset_id}",
            output_path=output_path,
            docs_base_url=docs_base_url,
            decision_record=decision.to_dict(),
        )
    return generate_eda_report(
        asset,
        time,
        title=title or "TSDataForge EDA Report",
        output_path=output_path,
        docs_base_url=docs_base_url,
        channel_names=channel_names,
    )


def handoff(
    source: Any,
    time: Any | None = None,
    *,
    output_dir: str | Path | None = "handoff_bundle",
    goal: str | None = None,
    include_report: bool = True,
    include_docs_site: bool = False,
    include_source_asset: bool = True,
    include_schemas: bool = True,
    dataset_id: str | None = None,
) -> DatasetHandoffBundle:
    asset = source if isinstance(source, (SeriesDataset, TaskDataset)) else coerce_asset(source, time, dataset_id=dataset_id)
    return build_dataset_handoff_bundle(
        asset,
        output_dir=output_dir,
        goal=goal,
        include_report=include_report,
        include_docs_site=include_docs_site,
        include_source_asset=include_source_asset,
        include_schemas=include_schemas,
    )


def taskify(
    source: Any,
    task: str,
    time: Any | None = None,
    *,
    dataset_id: str | None = None,
    **kwargs: Any,
) -> TaskDataset:
    asset = source if isinstance(source, SeriesDataset) else coerce_asset(source, time, dataset_id=dataset_id)
    if not isinstance(asset, SeriesDataset):
        raise TypeError("`taskify()` expects a base SeriesDataset or raw arrays/path that can be loaded into one.")
    return taskify_dataset(asset, task=task, **kwargs)


def demo(
    *,
    output_dir: str | Path = "demo_bundle",
    include_docs_site: bool = False,
    n_series: int = 24,
    length: int = 192,
    seed: int = 0,
    include_schemas: bool = True,
    scenario: str = "ecg_public",
) -> DatasetHandoffBundle:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    from .demo_assets import build_demo_dataset

    base = build_demo_dataset(
        scenario=scenario,
        n_series=int(n_series),
        length=int(length),
        seed=int(seed),
    )
    np.save(out / "demo_input.npy", base.values())
    return build_dataset_handoff_bundle(
        base,
        output_dir=out,
        include_report=True,
        include_docs_site=include_docs_site,
        goal=f"demo the shortest TSDataForge dataset -> report -> handoff path ({scenario})",
        include_schemas=include_schemas,
    )


def launch_gui(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    output_root: str | Path = ".bundle/gui_runs",
    open_browser: bool = True,
):
    from .gui import launch_gui as _launch_gui

    return _launch_gui(host=host, port=port, output_root=output_root, open_browser=open_browser)


__all__ = [
    "PublicEntrypoint",
    "PublicSurface",
    "public_surface",
    "render_public_surface_markdown",
    "save_public_surface",
    "load_asset",
    "report",
    "handoff",
    "taskify",
    "demo",
    "launch_gui",
]
