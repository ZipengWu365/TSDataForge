from pathlib import Path

from tsdataforge import generate_dataset, generate_series, generate_series_dataset
from tsdataforge.agent import (
    build_api_reference,
    build_dataset_context,
    build_series_context,
    build_task_context,
    build_task_dataset_card,
    example_catalog,
    generate_docs_site,
    recommend_examples,
    render_api_reference_markdown,
)
from tsdataforge.primitives import LinearTrend, MultiSineSeasonality, WhiteGaussianNoise



def test_build_series_context_pack():
    sample = generate_series(
        length=256,
        components=[
            LinearTrend(slope=0.02),
            MultiSineSeasonality(freqs=(24.0, 48.0), amps=(1.0, 0.2)),
            WhiteGaussianNoise(std=0.1),
        ],
        seed=0,
    )
    pack = build_series_context(sample, budget="small", goal="prepare forecasting baseline")
    assert pack.kind == "series"
    assert pack.estimated_tokens > 0
    assert "forecasting" in pack.compact["recommended_tasks"]
    assert len(pack.example_ids) >= 1



def test_build_dataset_and_task_contexts():
    base = generate_series_dataset(structures=["trend_seasonal_noise", "causal_varx"], n_series=10, length=96, seed=0)
    pack = build_dataset_context(base, goal="profile dataset for multiple tasks")
    assert pack.kind == "dataset"
    assert pack.compact["n_series"] == 10
    assert len(pack.compact["top_tags"]) >= 1

    task = base.taskify(task="forecasting", horizon=16)
    tpack = build_task_context(task)
    assert tpack.kind == "task"
    assert tpack.compact["task"] == "forecasting"
    assert tpack.compact["X_shape"][0] == 10



def test_cards_and_save_sidecars(tmp_path: Path):
    ds = generate_dataset(
        task="forecasting",
        structures=["trend_seasonal_noise"],
        n_series=6,
        length=80,
        horizon=12,
        seed=3,
    )
    card = build_task_dataset_card(ds)
    assert "forecasting" in card.to_markdown().lower()

    out = tmp_path / "saved_task"
    ds.save(out)
    assert (out / "task_card.json").exists()
    assert (out / "task_context.json").exists()
    assert (out / "README.md").exists()

    base = generate_series_dataset(structures=["trend_seasonal_noise"], n_series=4, length=64, seed=1)
    out2 = tmp_path / "saved_base"
    base.save(out2)
    assert (out2 / "dataset_card.json").exists()
    assert (out2 / "dataset_context.json").exists()
    assert (out2 / "README.md").exists()



def test_example_catalog_and_docs_site(tmp_path: Path):
    catalog = example_catalog()
    assert len(catalog) >= 20

    recs = recommend_examples("agent compact context forecasting api", top_k=4)
    assert len(recs) == 4
    assert any(ex.example_id == "agent_context_pack" for ex in recs)
    assert any(ex.example_id == "api_reference_overview" for ex in recs)

    site = generate_docs_site(tmp_path / "docs_site")
    assert len(site.pages) >= 8
    assert len(site.example_pages) >= 20
    assert len(site.api_pages) >= 5
    assert (tmp_path / "docs_site" / "index.html").exists()
    assert (tmp_path / "docs_site" / "cookbook.html").exists()
    assert (tmp_path / "docs_site" / "api-reference.html").exists()
    assert (tmp_path / "docs_site" / "faq.html").exists()
    assert (tmp_path / "docs_site" / "search-index.json").exists()
    assert (tmp_path / "docs_site" / "api-manifest.json").exists()



def test_api_reference_manifest_and_markdown():
    ref = build_api_reference()
    names = [sym.name for cat in ref.categories for sym in cat.symbols]
    assert ref.n_symbols <= 30
    assert "load_asset" in names
    assert "handoff" in names
    assert "report" in names
    assert "build_api_reference" in names
    md = render_api_reference_markdown(ref)
    assert "TSDataForge API Reference" in md
    assert "load_asset" in md

    full = build_api_reference(mode="full")
    full_names = [sym.name for cat in full.categories for sym in cat.symbols]
    assert full.n_symbols > ref.n_symbols
    assert "generate_series" in full_names
