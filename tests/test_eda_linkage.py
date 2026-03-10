from pathlib import Path

import numpy as np

from tsdataforge import (
    build_eda_resource_hub,
    common_eda_finding_routes,
    describe_series,
    generate_docs_site,
    generate_eda_report,
    generate_linked_dataset_eda_bundle,
    generate_linked_eda_bundle,
)
from tsdataforge.agent import render_eda_resource_hub_markdown



def test_build_eda_resource_hub_for_series():
    t = np.arange(256, dtype=float)
    y = 0.01 * t + np.sin(2 * np.pi * t / 24.0)
    desc = describe_series(y, t)
    hub = build_eda_resource_hub(desc, docs_base_url="docs/")
    assert hub.kind == "series"
    assert len(hub.page_links) >= 3
    assert any(link.path.startswith("docs/") for link in hub.page_links)
    assert any(item.task == "forecasting" for item in hub.recommended_tasks)
    assert len(hub.example_links) >= 1
    assert len(hub.api_links) >= 1
    assert len(hub.faq_links) >= 1
    md = render_eda_resource_hub_markdown(hub)
    assert "Recommended tasks" in md
    assert "Docs pages" in md



def test_generate_eda_report_contains_linked_resources(tmp_path: Path):
    t = np.arange(192, dtype=float)
    y = np.sin(2 * np.pi * t / 24.0) + 0.1 * np.random.default_rng(0).normal(size=len(t))
    out = tmp_path / "report.html"
    report = generate_eda_report(y, t, output_path=out, docs_base_url="docs/")
    assert out.exists()
    assert report.resource_hub is not None
    assert "Next actions and useful links" in report.html
    assert "Technical appendix" in report.html
    assert "docs/" in report.html
    assert (tmp_path / "report_resource_hub.json").exists()
    assert (tmp_path / "report_resource_hub.md").exists()



def test_generate_linked_eda_bundles_and_docs_site(tmp_path: Path):
    t = np.arange(160, dtype=float)
    y = np.sin(2 * np.pi * t / 16.0)
    bundle = generate_linked_eda_bundle(y, t, output_dir=tmp_path / "bundle")
    assert Path(bundle["report"]).exists()
    assert Path(bundle["docs_index"]).exists()
    assert Path(bundle["bundle_manifest"]).exists()
    assert (tmp_path / "bundle" / "report_resource_hub.json").exists()

    X = np.random.default_rng(0).normal(size=(12, 96, 3))
    dbundle = generate_linked_dataset_eda_bundle(X, output_dir=tmp_path / "dataset_bundle")
    assert Path(dbundle["report"]).exists()
    assert Path(dbundle["docs_index"]).exists()

    site = generate_docs_site(tmp_path / "site")
    assert site.eda_route_map is not None
    assert Path(site.eda_route_map).exists()
    assert (tmp_path / "site" / "faq.html").read_text(encoding="utf-8").find("eda_after_report") != -1
    assert len(common_eda_finding_routes()) >= 5
