from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsdataforge import demo, handoff, load_asset, public_surface, report, taskify
from tsdataforge.agent import build_api_reference
from tsdataforge.agent.schemas import build_artifact_schemas


def test_public_surface_lists_five_entrypoints():
    surface = public_surface()
    names = [item.name for item in surface.entrypoints]
    assert names == ["load_asset", "report", "handoff", "taskify", "demo"]


def test_load_report_handoff_and_taskify_with_arrays(tmp_path: Path):
    values = np.random.default_rng(0).normal(size=(8, 96))
    asset = load_asset(values, dataset_id="lab_values")
    rep = report(asset, output_path=tmp_path / "report.html")
    assert rep.output_path and rep.output_path.endswith("report.html")

    bundle = handoff(asset, output_dir=tmp_path / "handoff_bundle", include_report=True, include_schemas=True)
    assert (tmp_path / "handoff_bundle" / "report.html").exists()
    assert (tmp_path / "handoff_bundle" / "handoff_index_min.json").exists()
    assert (tmp_path / "handoff_bundle" / "action_plan.json").exists()
    assert (tmp_path / "handoff_bundle" / "schemas" / "handoff_index_min.schema.json").exists()

    forecast = taskify(asset, task="forecasting", horizon=12)
    assert forecast.task == "forecasting"
    assert len(forecast.X) == 8


def test_demo_and_public_vs_full_api_reference(tmp_path: Path):
    pytest.importorskip("matplotlib")
    bundle = demo(output_dir=tmp_path / "demo_bundle", include_schemas=True)
    assert bundle.output_dir is not None
    assert (tmp_path / "demo_bundle" / "demo_input.npy").exists()
    assert (tmp_path / "demo_bundle" / "schemas" / "schema_catalog.json").exists()
    assert (tmp_path / "demo_bundle" / "schemas" / "tool_contracts.json").exists()

    public_ref = build_api_reference()
    full_ref = build_api_reference(mode="full")
    assert public_ref.n_symbols < full_ref.n_symbols


def test_build_artifact_schemas_catalog():
    catalog = build_artifact_schemas()
    ids = [item.schema_id for item in catalog.schemas]
    assert ids == [
        "dataset_context",
        "dataset_card",
        "handoff_index_min",
        "handoff_index",
        "action_plan",
        "handoff_bundle",
    ]
