from pathlib import Path

from tsdataforge import (
    build_api_reference,
    environment_catalog,
    generate_docs_site,
    recommend_environments,
    recommend_scenarios,
    scenario_catalog,
)


def test_scenario_and_environment_catalogs_are_available():
    scenarios = scenario_catalog(language="en")
    envs = environment_catalog(language="en")
    assert len(scenarios) >= 6
    assert len(envs) >= 5

    rec_scenarios = recommend_scenarios("real data report task selection", top_k=2, language="en")
    assert len(rec_scenarios) == 2
    assert any(item.scenario_id == "real-data-profile-and-taskify" for item in rec_scenarios)

    rec_envs = recommend_environments("agent compact context docs", top_k=2, language="en")
    assert len(rec_envs) == 2
    assert any(item.env_id == "llm-agent-workflow" for item in rec_envs)



def test_api_reference_has_rationale_and_environment_fields():
    ref = build_api_reference()
    names = {sym.name: sym for cat in ref.categories for sym in cat.symbols}
    assert names["load_asset"].why_exists
    assert names["load_asset"].works_in
    assert names["generate_eda_report"].scenario_ids
    md = ref.to_markdown()
    assert "Why it exists" in md
    assert "Works well in" in md



def test_docs_site_explains_package_and_routes_by_scenario(tmp_path: Path):
    site = generate_docs_site(tmp_path / "site")
    index_html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    use_cases_html = (tmp_path / "site" / "use-cases.html").read_text(encoding="utf-8")
    api_html = (tmp_path / "site" / "api-reference.html").read_text(encoding="utf-8")
    faq_html = (tmp_path / "site" / "faq.html").read_text(encoding="utf-8")

    assert "What TSDataForge is" in index_html
    assert "What it is not" in index_html
    assert "Which environment should I use" in index_html
    assert "Which APIs matter in which scenario" in api_html
    assert "What is TSDataForge actually for?" in faq_html
    assert "What people actually use the package for" in use_cases_html
    assert site.eda_route_map is not None
