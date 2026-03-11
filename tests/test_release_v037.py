from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsdataforge import build_artifact_schemas, build_tool_contracts, demo, handoff


def test_min_index_and_action_plan_are_saved(tmp_path: Path):
    pytest.importorskip("matplotlib")
    bundle = demo(output_dir=tmp_path / "demo_bundle", scenario="ecg_public")
    min_payload = json.loads((tmp_path / "demo_bundle" / "handoff_index_min.json").read_text(encoding="utf-8"))
    plan_payload = json.loads((tmp_path / "demo_bundle" / "action_plan.json").read_text(encoding="utf-8"))
    assert (tmp_path / "demo_bundle" / "decision_record.json").exists()
    assert (tmp_path / "demo_bundle" / "decision_record.md").exists()
    assert min_payload["agent_entrypoint"] == "handoff_index_min.json"
    assert min_payload["action_plan_path"] == "action_plan.json"
    assert min_payload["decision_path"] == str(tmp_path / "demo_bundle" / "decision_record.json")
    assert plan_payload["recommended_next_step"] == bundle.index.recommended_next_step
    assert plan_payload["decision_path"] == str(tmp_path / "demo_bundle" / "decision_record.json")


def test_min_index_is_smaller_than_full_index(tmp_path: Path):
    pytest.importorskip("matplotlib")
    bundle = demo(output_dir=tmp_path / "demo_bundle", scenario="macro_public")
    min_len = len((tmp_path / "demo_bundle" / "handoff_index_min.json").read_text(encoding="utf-8"))
    full_len = len((tmp_path / "demo_bundle" / "handoff_index.json").read_text(encoding="utf-8"))
    assert min_len < full_len


def test_tool_contracts_and_schema_catalog_cover_public_surface():
    tools = build_tool_contracts().tools
    schemas = build_artifact_schemas().schemas
    assert [tool.tool_id for tool in tools] == ["load_asset", "report", "handoff", "taskify", "demo"]
    assert {schema.schema_id for schema in schemas} >= {"handoff_index_min", "action_plan", "handoff_bundle"}
