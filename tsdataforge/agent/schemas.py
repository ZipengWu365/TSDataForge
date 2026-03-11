from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactSchema:
    schema_id: str
    title: str
    summary: str
    schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactSchemaCatalog:
    version: str
    schemas: tuple[ArtifactSchema, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "schemas": [item.to_dict() for item in self.schemas]}


def build_artifact_schemas(version: str = "0.3.7") -> ArtifactSchemaCatalog:
    schemas = (
        ArtifactSchema(
            schema_id="dataset_context",
            title="Dataset context",
            summary="Compact, low-token summary of a base dataset for agent or teammate handoff.",
            schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["kind", "budget", "compact", "next_actions"],
                "properties": {
                    "kind": {"const": "dataset"},
                    "budget": {"type": "string"},
                    "compact": {"type": "object"},
                    "narrative": {"type": "array", "items": {"type": "string"}},
                    "example_ids": {"type": "array", "items": {"type": "string"}},
                    "next_actions": {"type": "array", "items": {"type": "string"}},
                    "estimated_tokens": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": True,
            },
        ),
        ArtifactSchema(
            schema_id="dataset_card",
            title="Dataset card",
            summary="Human-readable dataset summary that travels with the asset.",
            schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["kind", "title", "summary", "sections"],
                "properties": {
                    "kind": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "sections": {"type": "array", "items": {"type": "object"}},
                },
                "additionalProperties": True,
            },
        ),
        ArtifactSchema(
            schema_id="decision_record",
            title="Decision record",
            summary="Explicit routing logic: key facts, risks, candidate tasks, blockers, and one recommended next step.",
            schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["kind", "dataset_id", "summary", "facts", "risks", "blockers", "candidate_tasks", "recommended_next_step"],
                "properties": {
                    "kind": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "goal": {"type": ["string", "null"]},
                    "summary": {"type": "string"},
                    "facts": {"type": "array", "items": {"type": "object"}},
                    "risks": {"type": "array", "items": {"type": "object"}},
                    "blockers": {"type": "array", "items": {"type": "object"}},
                    "candidate_tasks": {"type": "array", "items": {"type": "object"}},
                    "recommended_next_step": {"type": ["object", "null"]},
                },
                "additionalProperties": False,
            },
        ),
        ArtifactSchema(
            schema_id="handoff_index_min",
            title="Minimal handoff index",
            summary="Tiny first-entry contract for agents: what this bundle is, what to open next, and the single recommended action.",
            schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["kind", "dataset_id", "title", "what_this_is", "agent_open_order", "recommended_next_step"],
                "properties": {
                    "kind": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "title": {"type": "string"},
                    "wow_sentence": {"type": "string"},
                    "what_this_is": {"type": "string"},
                    "agent_entrypoint": {"const": "handoff_index_min.json"},
                    "agent_open_order": {"type": "array", "items": {"type": "string"}},
                    "action_plan_path": {"type": "string"},
                    "recommended_next_step": {"type": ["string", "null"]},
                    "why_recommended": {"type": ["string", "null"]},
                    "decision_path": {"type": ["string", "null"]},
                    "decision_summary": {"type": ["string", "null"]},
                    "main_risks": {"type": "array", "items": {"type": "string"}},
                    "top_candidate_tasks": {"type": "array", "items": {"type": "string"}},
                    "decision_confidence": {"type": ["number", "null"]},
                    "recommended_prompt": {"type": "string"},
                },
                "additionalProperties": False,
            },
        ),
        ArtifactSchema(
            schema_id="handoff_index",
            title="Handoff index",
            summary="Compact routing map for human and agent open order plus the single recommended next step.",
            schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["kind", "dataset_id", "title", "summary", "human_open_order", "agent_open_order", "recommended_next_step"],
                "properties": {
                    "kind": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "wow_sentence": {"type": "string"},
                    "primary_open_order": {"type": "array", "items": {"type": "string"}},
                    "human_open_order": {"type": "array", "items": {"type": "string"}},
                    "agent_entrypoint": {"const": "handoff_index_min.json"},
                    "agent_open_order": {"type": "array", "items": {"type": "string"}},
                    "action_plan_path": {"type": ["string", "null"]},
                    "report_path": {"type": ["string", "null"]},
                    "card_path": {"type": ["string", "null"]},
                    "context_path": {"type": ["string", "null"]},
                    "docs_index": {"type": ["string", "null"]},
                    "recommended_next_step": {"type": ["string", "null"]},
                    "why_recommended": {"type": ["string", "null"]},
                    "decision_path": {"type": ["string", "null"]},
                    "decision_summary": {"type": ["string", "null"]},
                    "main_risks": {"type": "array", "items": {"type": "string"}},
                    "top_candidate_tasks": {"type": "array", "items": {"type": "string"}},
                    "decision_confidence": {"type": ["number", "null"]},
                    "recommended_prompt": {"type": "string"},
                    "action_status_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
                },
                "additionalProperties": False,
            },
        ),
        ArtifactSchema(
            schema_id="action_plan",
            title="Action plan",
            summary="Detailed already_done / recommended / optional plan that expands the compact handoff index.",
            schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["kind", "dataset_id", "action_plan"],
                "properties": {
                    "kind": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "recommended_next_step": {"type": ["string", "null"]},
                    "already_done_actions": {"type": "array", "items": {"type": "string"}},
                    "action_status_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
                    "decision_path": {"type": ["string", "null"]},
                    "action_plan": {"type": "array", "items": {"type": "object"}},
                },
                "additionalProperties": False,
            },
        ),
        ArtifactSchema(
            schema_id="handoff_bundle",
            title="Handoff bundle inventory",
            summary="Compact inventory of the bundle, previews of the main artifacts, and a structured action plan.",
            schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["kind", "dataset_id", "title", "summary", "open_order", "next_actions", "artifacts", "action_plan"],
                "properties": {
                    "kind": {"type": "string"},
                    "dataset_id": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "open_order": {"type": "array", "items": {"type": "string"}},
                    "next_actions": {"type": "array", "items": {"type": "string"}},
                    "artifacts": {"type": "array", "items": {"type": "object"}},
                    "action_plan": {"type": "array", "items": {"type": "object"}},
                    "context_preview": {"type": "object"},
                    "card_preview": {"type": "object"},
                    "decision_record": {"type": "object"},
                    "handoff_index": {"type": "object"},
                },
                "additionalProperties": True,
            },
        ),
    )
    return ArtifactSchemaCatalog(version=version, schemas=schemas)


def save_artifact_schemas(catalog: ArtifactSchemaCatalog | None, output_dir: str | Path) -> ArtifactSchemaCatalog:
    cat = catalog or build_artifact_schemas()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for item in cat.schemas:
        (out / f"{item.schema_id}.schema.json").write_text(json.dumps(item.schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "schema_catalog.json").write_text(json.dumps(cat.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return cat


__all__ = ["ArtifactSchema", "ArtifactSchemaCatalog", "build_artifact_schemas", "save_artifact_schemas"]
