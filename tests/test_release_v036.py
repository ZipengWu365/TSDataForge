from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsdataforge import demo, handoff, load_asset
from tsdataforge.cli import build_parser, main as cli_main
from tsdataforge.demo_assets import build_demo_dataset, demo_scenario_catalog


@pytest.mark.parametrize("scenario", ["synthetic", "ecg_public", "macro_public", "climate_public", "sunspots_public", "icu_vitals"])
def test_demo_scenarios_build_base_datasets(scenario: str):
    dataset = build_demo_dataset(scenario=scenario, n_series=6, length=96, seed=0)
    assert dataset.dataset_id.endswith("demo") or dataset.dataset_id == "series_dataset"
    assert len(dataset.series) >= 6


def test_demo_and_handoff_default_to_schema_first(tmp_path: Path):
    pytest.importorskip("matplotlib")
    values = load_asset(__import__("numpy").random.default_rng(0).normal(size=(6, 96)), dataset_id="lab_values")
    bundle = handoff(values, output_dir=tmp_path / "handoff_bundle")
    assert bundle.index is not None
    assert (tmp_path / "handoff_bundle" / "schemas" / "schema_catalog.json").exists()
    assert (tmp_path / "handoff_bundle" / "schemas" / "tool_contracts.json").exists()
    assert bundle.index.recommended_next_step is not None
    assert bundle.index.recommended_next_step != "run:build_dataset_handoff_bundle"

    demo_bundle = demo(output_dir=tmp_path / "demo_bundle")
    assert (tmp_path / "demo_bundle" / "schemas" / "schema_catalog.json").exists()
    assert demo_bundle.index is not None
    assert demo_bundle.index.recommended_next_step != "run:build_dataset_handoff_bundle"


def test_cli_help_and_demo_summary_are_clean(tmp_path: Path, capsys):
    parser = build_parser()
    help_text = parser.format_help()
    assert "�" not in help_text
    assert "report and handoff workflows for time-series dataset assets" in help_text

    pytest.importorskip("matplotlib")
    code = cli_main(["demo", "--scenario", "ecg_public", "--output", str(tmp_path / "ecg_bundle"), "--n-series", "8", "--length", "256"])
    assert code == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["agent_entrypoint"] == "handoff_index_min.json"
    assert payload["human_open_order"][0] == "report.html"
    assert payload["agent_open_order"][0] == "dataset_context.json"
    assert payload["recommended_next_step"]


def test_docs_and_release_surface_files_exist():
    assert Path(".github/workflows/release.yml").exists()
    assert Path(".github/workflows/docs.yml").exists()
    assert Path(".github/ISSUE_TEMPLATE/bug_report.yml").exists()
    assert Path("CITATION.cff").exists()
    assert Path("SECURITY.md").exists()


def test_readme_mentions_real_world_scenarios_and_open_order():
    readme = Path("README.md").read_text(encoding="utf-8")
    quickstart = Path("docs/quickstart.md").read_text(encoding="utf-8")
    handoff_md = Path("docs/handoff.md").read_text(encoding="utf-8")
    assert "Public ECG arrhythmia" in readme
    assert "Public US macro" in readme
    assert "Public climate CO₂" in readme
    assert "Open these files in this order" in readme
    assert "agent open order" in handoff_md.lower()
    assert "30-second path" in quickstart


def test_demo_scenario_catalog_contains_flagship_cases():
    titles = [item.title for item in demo_scenario_catalog(language="en")]
    assert any("ECG" in title for title in titles)
    assert any("macro" in title.lower() for title in titles)
    assert any("climate" in title.lower() for title in titles)
