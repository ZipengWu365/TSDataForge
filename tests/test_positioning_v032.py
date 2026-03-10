from pathlib import Path

from tsdataforge import (
    build_positioning_matrix,
    competitor_catalog,
    generate_docs_site,
    recommend_companions,
    render_positioning_markdown,
    save_positioning_matrix,
)


def test_positioning_matrix_contains_expected_libraries(tmp_path):
    matrix = build_positioning_matrix()
    ids = [item.package_id for item in matrix.profiles]
    assert "tsdataforge" in ids
    assert "tsfresh" in ids
    assert "sktime" in ids
    assert "darts" in ids
    assert "tslearn" in ids

    md = render_positioning_markdown(matrix)
    assert "How TSDataForge differs" in md
    assert "tsfresh" in md.lower()

    md_path = tmp_path / "positioning.md"
    json_path = tmp_path / "positioning.json"
    save_positioning_matrix(matrix, md_path)
    save_positioning_matrix(matrix, json_path)
    assert md_path.exists() and json_path.exists()
    assert "TSDataForge ecosystem positioning" in md_path.read_text(encoding="utf-8")


def test_recommend_companions_is_goal_sensitive():
    feature_tools = [item.package_id for item in recommend_companions("feature extraction classification", top_k=3)]
    assert "tsfresh" in feature_tools

    forecast_tools = [item.package_id for item in recommend_companions("probabilistic forecasting deep learning", top_k=3)]
    assert "gluonts" in forecast_tools or "darts" in forecast_tools


def test_docs_site_contains_positioning_page(tmp_path):
    site = generate_docs_site(tmp_path / "site")
    assert any(path.endswith("positioning.html") for path in site.pages)
    positioning = (tmp_path / "site" / "positioning.html").read_text(encoding="utf-8")
    assert "How TSDataForge differs from adjacent libraries" in positioning
    index_html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "Why this is not just another time-series package" in index_html
    faq_html = (tmp_path / "site" / "faq.html").read_text(encoding="utf-8")
    assert "Why not just use sktime, Darts, tsfresh" in faq_html
