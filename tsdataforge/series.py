from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .specs import SeriesSpec
from .trace import SeriesTrace


@dataclass
class GeneratedSeries:
    time: np.ndarray
    values: np.ndarray
    spec: SeriesSpec
    trace: SeriesTrace | None = None

    def save(self, directory: str | Path, *, include_card: bool = True, include_agent_context: bool = True) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "series.npz",
            time=self.time,
            values=self.values,
            latent=None if self.trace is None else self.trace.latent,
            observed=self.values,
        )
        metadata = {
            "spec": self.spec.to_dict(),
            "has_trace": self.trace is not None,
        }
        if self.trace is not None:
            metadata["trace"] = self.trace.to_metadata()
        with (directory / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        if include_agent_context:
            from .agent import build_series_context

            pack = build_series_context(self, budget="small")
            (directory / "series_context.json").write_text(
                json.dumps(pack.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (directory / "series_context.md").write_text(pack.to_markdown(), encoding="utf-8")

        if include_card:
            title = f"GeneratedSeries card: {self.spec.structure_id or self.spec.name or 'series'}"
            readme = [
                f"# {title}",
                "",
                f"- shape: {tuple(np.asarray(self.values).shape)}",
                f"- structure_id: {self.spec.structure_id or self.spec.name or 'unknown'}",
                f"- has_trace: {self.trace is not None}",
                "",
                "Open `series_context.md` for the compact agent-friendly summary.",
            ]
            (directory / "README.md").write_text("\n".join(readme), encoding="utf-8")
