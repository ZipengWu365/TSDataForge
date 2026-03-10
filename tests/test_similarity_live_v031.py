import numpy as np

from tsdataforge import (
    compare_series,
    find_top_matches,
    fetch_fred_series,
    fetch_github_stars_series,
    generate_docs_site,
    pairwise_similarity,
)


def test_compare_series_and_pairwise_similarity_work():
    t = np.arange(256, dtype=float)
    a = np.sin(2 * np.pi * t / 24.0)
    b = np.sin(2 * np.pi * (t + 2.0) / 24.0) + 0.05 * np.random.default_rng(0).normal(size=len(t))
    c = np.random.default_rng(1).normal(size=len(t))

    res_ab = compare_series(a, b, reference_name="a", candidate_name="b")
    res_ac = compare_series(a, c, reference_name="a", candidate_name="c")
    assert res_ab.aggregate_score > res_ac.aggregate_score
    assert res_ab.aggregate_score > 0.65

    matrix = pairwise_similarity({"a": a, "b": b, "c": c})
    assert matrix.matrix.shape == (3, 3)
    md = matrix.to_markdown()
    assert "a" in md and "b" in md and "c" in md

    matches = find_top_matches(a, {"b": b, "c": c}, reference_name="a")
    assert matches[0].candidate_name == "b"


def test_fetch_fred_series_with_monkeypatched_request(monkeypatch):
    from tsdataforge.integrations import live

    def fake_request_json(url, *, headers=None, params=None):
        return {
            "observations": [
                {"date": "2026-01-01", "value": "10.0"},
                {"date": "2026-01-02", "value": "11.0"},
                {"date": "2026-01-03", "value": "."},
            ]
        }

    monkeypatch.setattr(live, "_request_json", fake_request_json)
    series = fetch_fred_series("TESTSERIES", api_key="demo")
    assert series.values.shape == (2,)
    assert float(series.values[-1]) == 11.0
    assert series.trace is not None
    assert series.trace.states["external/fred/series_id"] == "TESTSERIES"


def test_fetch_github_stars_series_rest_with_monkeypatched_request(monkeypatch):
    from tsdataforge.integrations import live

    def fake_request_json(url, *, headers=None, params=None):
        page = int((params or {}).get("page", 1))
        if page == 1:
            return [
                {"starred_at": "2026-01-01T10:00:00Z"},
                {"starred_at": "2026-01-01T11:00:00Z"},
            ]
        if page == 2:
            return [
                {"starred_at": "2026-01-02T12:00:00Z"},
            ]
        return []

    monkeypatch.setattr(live, "_request_json", fake_request_json)
    series = fetch_github_stars_series("foo", "bar", mode="rest", max_pages=5, per_page=2)
    assert series.values.shape[0] == 2
    assert np.allclose(series.values, np.array([2.0, 3.0]))
    assert series.trace is not None
    daily = np.asarray(series.trace.states["external/github/daily_new_stars"])
    assert np.allclose(daily, np.array([2.0, 1.0]))


def test_docs_site_surfaces_hot_right_now_examples(tmp_path):
    generate_docs_site(tmp_path / "site")
    index_html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "Hot right now" in index_html
    assert "openclaw_stars_similarity" in index_html
    example_html = (tmp_path / "site" / "examples" / "openclaw_stars_similarity.html").read_text(encoding="utf-8")
    assert "GitHub" in example_html or "OpenClaw" in example_html
