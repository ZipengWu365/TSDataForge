from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .eda import generate_dataset_eda_report, generate_eda_report


def generate_linked_eda_bundle(
    values: Any,
    time: np.ndarray | None = None,
    *,
    output_dir: str | Path = "tsdataforge_eda_bundle",
    title: str = "TSDataForge Linked EDA Report",
    docs_title: str = "TSDataForge Docs",
    channel_names: list[str] | None = None,
    include_suggested_spec: bool = True,
) -> dict[str, str]:
    """Generate a shareable bundle with a linked EDA report and docs site."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs_dir = out / "docs"
    from ..agent.site import generate_docs_site

    site = generate_docs_site(docs_dir, title=docs_title)
    generate_eda_report(
        values,
        time,
        channel_names=channel_names,
        title=title,
        output_path=out / "report.html",
        include_suggested_spec=include_suggested_spec,
        docs_base_url="docs/",
        include_linked_resources=True,
    )
    manifest = {
        "report": str(out / "report.html"),
        "docs_index": str(docs_dir / "index.html"),
        "docs_site": site.to_dict(),
    }
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# TSDataForge Linked EDA Bundle\n\n"
        "- Open `report.html` for the human-readable report.\n"
        "- Use links inside the report to jump to `docs/` pages, examples, API, and FAQ.\n"
        "- Share this whole folder as a portable bundle.\n",
        encoding="utf-8",
    )
    return {
        "report": str(out / "report.html"),
        "docs_index": str(docs_dir / "index.html"),
        "bundle_manifest": str(out / "bundle_manifest.json"),
    }


def generate_linked_dataset_eda_bundle(
    values: Any,
    time: Any | None = None,
    *,
    output_dir: str | Path = "tsdataforge_dataset_eda_bundle",
    title: str = "TSDataForge Linked Dataset EDA Report",
    docs_title: str = "TSDataForge Docs",
    max_series: int | None = 200,
    seed: int = 0,
) -> dict[str, str]:
    """Generate a shareable dataset-level bundle with report + docs site."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs_dir = out / "docs"
    from ..agent.site import generate_docs_site

    site = generate_docs_site(docs_dir, title=docs_title)
    generate_dataset_eda_report(
        values,
        time,
        title=title,
        output_path=out / "report.html",
        max_series=max_series,
        seed=seed,
        docs_base_url="docs/",
        include_linked_resources=True,
    )
    manifest = {
        "report": str(out / "report.html"),
        "docs_index": str(docs_dir / "index.html"),
        "docs_site": site.to_dict(),
    }
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# TSDataForge Linked Dataset EDA Bundle\n\n"
        "- Open `report.html` for the dataset-level report.\n"
        "- Use links inside the report to jump to `docs/` pages, examples, API, and FAQ.\n"
        "- Share this whole folder as a portable bundle.\n",
        encoding="utf-8",
    )
    return {
        "report": str(out / "report.html"),
        "docs_index": str(docs_dir / "index.html"),
        "bundle_manifest": str(out / "bundle_manifest.json"),
    }
