from pathlib import Path

import numpy as np
import pytest

from tsdataforge import build_dataset_handoff_bundle, generate_series_dataset
from tsdataforge.cli import main as cli_main
from tsdataforge.agent import build_api_reference, generate_docs_site


def test_build_dataset_handoff_bundle(tmp_path: Path):
    pytest.importorskip("matplotlib")

    base = generate_series_dataset(
        structures=["trend_seasonal_noise", "regime_switch"],
        n_series=10,
        length=96,
        seed=0,
    )
    bundle = build_dataset_handoff_bundle(base, output_dir=tmp_path / "bundle", include_report=True)
    assert bundle.dataset_id.startswith("synthetic_")
    assert bundle.next_actions[0] == "open:report.html"
    assert (tmp_path / "bundle" / "report.html").exists()
    assert (tmp_path / "bundle" / "decision_record.json").exists()
    assert (tmp_path / "bundle" / "dataset_card.md").exists()
    assert (tmp_path / "bundle" / "dataset_context.json").exists()
    assert (tmp_path / "bundle" / "handoff_bundle.json").exists()
    assert any(item.name == "report.html" for item in bundle.artifacts)


def test_dataset_save_writes_handoff_bundle(tmp_path: Path):
    base = generate_series_dataset(structures=["trend_seasonal_noise"], n_series=4, length=64, seed=1)
    out = tmp_path / "saved_base"
    base.save(out)
    assert (out / "dataset_card.md").exists()
    assert (out / "handoff_bundle.json").exists()
    assert (out / "handoff_bundle.md").exists()


def test_cli_handoff_command(tmp_path: Path):
    pytest.importorskip("matplotlib")
    path = tmp_path / "demo.npy"
    np.save(path, np.random.default_rng(0).normal(size=(12, 80)))
    code = cli_main(["handoff", str(path), "--output", str(tmp_path / "handoff")])
    assert code == 0
    assert (tmp_path / "handoff" / "report.html").exists()
    assert (tmp_path / "handoff" / "dataset_context.json").exists()


def test_docs_and_api_surface_include_handoff(tmp_path: Path):
    site = generate_docs_site(tmp_path / "docs")
    assert (tmp_path / "docs" / "handoff.html").exists()
    ref = build_api_reference()
    names = [sym.name for cat in ref.categories for sym in cat.symbols]
    assert "build_dataset_handoff_bundle" in names
    assert "DatasetHandoffBundle" in names
