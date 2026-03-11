from pathlib import Path

from tsdataforge.agent import example_catalog, generate_docs_site, recommend_tutorials, tutorial_catalog


def test_examples_and_tutorials_are_localizable():
    en_examples = example_catalog(language="en")
    zh_examples = example_catalog(language="zh")
    assert en_examples[0].title != zh_examples[0].title
    assert "quickstart" in en_examples[0].title.lower()

    en_tutorials = tutorial_catalog(language="en")
    zh_tutorials = tutorial_catalog(language="zh")
    assert en_tutorials[0].title != zh_tutorials[0].title

    recs = recommend_tutorials("real data eda forecasting", top_k=2, language="en")
    assert len(recs) == 2
    assert any(item.tutorial_id == "real-data-to-first-model" for item in recs)


def test_docs_site_has_english_root_and_chinese_mirror(tmp_path: Path):
    site = generate_docs_site(tmp_path / "site")
    index = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    zh_index = (tmp_path / "site" / "zh" / "index.html").read_text(encoding="utf-8")
    assert "Quickstart" in index
    assert "zxw365@student.bham.ac.uk" in index
    assert "中文" in index
    assert "English" in zh_index
    assert "href='../index.html'" in zh_index
    assert (tmp_path / "site" / "tutorials.html").exists()
    assert (tmp_path / "site" / "use-cases.html").exists()
    assert (tmp_path / "site" / "zh" / "tutorials.html").exists()
    assert len(site.zh_pages) >= 8
