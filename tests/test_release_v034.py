from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsdataforge import build_dataset_handoff_bundle, generate_series_dataset
from tsdataforge.cli import main as cli_main


def test_handoff_indices_and_bundle_json_are_compact(tmp_path: Path):
    pytest.importorskip("matplotlib")
    base = generate_series_dataset(structures=["trend_seasonal_noise", "regime_switch"], n_series=8, length=96, seed=0)
    bundle = build_dataset_handoff_bundle(base, output_dir=tmp_path / "bundle", include_report=True)
    assert bundle.index is not None
    assert (tmp_path / "bundle" / "handoff_index_min.json").exists()
    assert (tmp_path / "bundle" / "handoff_index.json").exists()
    assert (tmp_path / "bundle" / "action_plan.json").exists()
    payload = json.loads((tmp_path / "bundle" / "handoff_bundle.json").read_text(encoding="utf-8"))
    assert "context" not in payload
    assert "card" not in payload
    assert payload["handoff_index"]["primary_open_order"][0] == "report.html"
    assert payload["handoff_index"]["context_path"].endswith("dataset_context.json")


def test_cli_demo_command_creates_visual_first_bundle(tmp_path: Path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "demo_bundle"
    code = cli_main(["demo", "--output", str(out), "--n-series", "12", "--length", "96"])
    assert code == 0
    assert (out / "report.html").exists()
    assert (out / "handoff_index_min.json").exists()
    assert (out / "demo_input.npy").exists()


def test_source_docs_and_readme_are_aligned():
    quickstart = Path("docs/quickstart.md").read_text(encoding="utf-8")
    handoff = Path("docs/handoff.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "python -m tsdataforge demo --output demo_bundle" in quickstart
    assert "handoff_index_min.json" in handoff
    assert "showcase/assets/raw-vs-bundle.svg" in readme
    assert "Public ECG arrhythmia handoff" in readme


def test_showcase_assets_exist():
    assert Path("showcase/README.md").exists()
    assert Path("showcase/assets/tsdataforge-demo-flow.gif").exists()
    assert Path("showcase/assets/raw-vs-bundle.svg").exists()
    assert Path("showcase/assets/report-preview.svg").exists()
    assert Path("showcase/assets/ecg-public-preview.svg").exists()
