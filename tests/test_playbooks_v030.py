from pathlib import Path

from tsdataforge import (
    create_starter_project,
    export_tutorial_notebooks,
    generate_docs_site,
    playbook_catalog,
    recommend_playbooks,
    recommend_starters,
    starter_catalog,
)



def test_playbook_and_starter_catalogs_are_recommendable():
    playbooks = playbook_catalog(language="en")
    starters = starter_catalog(language="en")
    assert len(playbooks) >= 5
    assert len(starters) >= 5

    rec_playbooks = recommend_playbooks("real data report task choice", top_k=2, language="en")
    assert len(rec_playbooks) == 2
    assert any(item.playbook_id == "real-data-understanding" for item in rec_playbooks)

    rec_starters = recommend_starters("benchmark multiple tasks reusable dataset", top_k=2, language="en")
    assert len(rec_starters) == 2
    assert any(item.starter_id == "benchmark-lab" for item in rec_starters)



def test_export_tutorial_notebooks_and_create_starter_project(tmp_path: Path):
    notebook_dir = tmp_path / "notebooks"
    assets = export_tutorial_notebooks(notebook_dir, tutorial_ids=["first-five-minutes"], language="en")
    assert len(assets) == 1
    assert (notebook_dir / "first-five-minutes.ipynb").exists()
    assert (notebook_dir / "first-five-minutes.py").exists()
    assert (notebook_dir / "notebook_manifest.json").exists()

    starter = create_starter_project(tmp_path / "starter_project", "first-success-notebook", language="en")
    assert Path(starter.readme_path).exists()
    assert Path(starter.manifest_path).exists()
    assert len(starter.script_paths) >= 2
    assert len(starter.notebook_paths) >= 2
    assert (tmp_path / "starter_project" / "requirements.txt").exists()



def test_docs_site_exports_playbooks_starters_and_notebooks(tmp_path: Path):
    site = generate_docs_site(tmp_path / "site")
    assert (tmp_path / "site" / "playbooks.html").exists()
    assert (tmp_path / "site" / "starter-kits.html").exists()
    assert (tmp_path / "site" / "notebooks" / "first-five-minutes.ipynb").exists()
    assert (tmp_path / "site" / "starters" / "first-success-notebook" / "README.md").exists()
    assert len(site.notebook_files) >= 2
    assert len(site.starter_projects) >= 5
    assert len(site.showcase_bundles) >= 5

    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "Workflows" in html
    assert "Starter Projects" in html
