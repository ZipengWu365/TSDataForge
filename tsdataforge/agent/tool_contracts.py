from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolContract:
    tool_id: str
    function_name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    input_example: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolContractCatalog:
    version: str
    tools: tuple[ToolContract, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "tools": [tool.to_dict() for tool in self.tools]}

    def to_markdown(self) -> str:
        lines = [
            f"# TSDataForge Tool Contracts v{self.version}",
            "",
            "Structured contracts for the five public entry points. These are designed for tool-calling agents, MCP-style wrappers, or thin service layers.",
            "",
        ]
        for tool in self.tools:
            lines.extend([
                f"## `{tool.tool_id}`",
                "",
                tool.description,
                "",
                "### Function",
                "",
                f"`{tool.function_name}`",
                "",
                "### Input schema",
                "",
                "```json",
                json.dumps(tool.input_schema, ensure_ascii=False, indent=2),
                "```",
                "",
                "### Output schema",
                "",
                "```json",
                json.dumps(tool.output_schema, ensure_ascii=False, indent=2),
                "```",
            ])
            if tool.input_example:
                lines.extend([
                    "",
                    "### Input example",
                    "",
                    "```json",
                    json.dumps(tool.input_example, ensure_ascii=False, indent=2),
                    "```",
                ])
            lines.append("")
        return "\n".join(lines).strip() + "\n"


def _artifact_output_schema(kind: str) -> dict[str, Any]:
    if kind == "report":
        return {
            "type": "object",
            "required": ["title", "output_path", "recommended_tasks"],
            "properties": {
                "title": {"type": "string"},
                "output_path": {"type": "string"},
                "recommended_tasks": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": True,
        }
    if kind == "handoff":
        return {
            "type": "object",
            "required": ["output_dir", "kind", "dataset_id"],
            "properties": {
                "output_dir": {"type": "string"},
                "kind": {"type": "string"},
                "dataset_id": {"type": "string"},
                "next_actions": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": True,
        }
    if kind == "taskify":
        return {
            "type": "object",
            "required": ["task", "schema"],
            "properties": {
                "task": {"type": "string"},
                "schema": {"type": "object"},
                "X": {"type": ["array", "object"]},
                "y": {"type": ["array", "object", "null"]},
            },
            "additionalProperties": True,
        }
    if kind == "load_asset":
        return {
            "type": "object",
            "required": ["dataset_id"],
            "properties": {
                "dataset_id": {"type": "string"},
                "channel_names": {"type": ["array", "null"], "items": {"type": "string"}},
            },
            "additionalProperties": True,
        }
    return {
        "type": "object",
        "required": ["output_dir"],
        "properties": {
            "output_dir": {"type": "string"},
            "recommended_next_step": {"type": ["string", "null"]},
            "agent_open_order": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    }


def build_tool_contracts(version: str = "0.3.7") -> ToolContractCatalog:
    base_object = {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }
    tools = (
        ToolContract(
            tool_id="load_asset",
            function_name="tsdataforge.load_asset",
            description="Load a path or raw arrays into a TSDataForge asset without forcing the caller to know file-type-specific loader code.",
            input_schema={
                **base_object,
                "required": ["source"],
                "properties": {
                    "source": {"type": ["string", "array", "object"]},
                    "time": {"type": ["array", "object", "null"]},
                    "dataset_id": {"type": ["string", "null"]},
                },
            },
            output_schema=_artifact_output_schema("load_asset"),
            input_example={"source": "my_dataset.npy", "dataset_id": "patient_monitor_v2"},
        ),
        ToolContract(
            tool_id="report",
            function_name="tsdataforge.report",
            description="Render the fastest human-readable HTML explanation for a series or dataset asset.",
            input_schema={
                **base_object,
                "required": ["source"],
                "properties": {
                    "source": {"type": ["string", "array", "object"]},
                    "time": {"type": ["array", "object", "null"]},
                    "output_path": {"type": ["string", "null"]},
                },
            },
            output_schema=_artifact_output_schema("report"),
            input_example={"source": "my_dataset.npy", "output_path": "report.html"},
        ),
        ToolContract(
            tool_id="handoff",
            function_name="tsdataforge.handoff",
            description="Build the shortest report + card + context + handoff surface for a dataset asset.",
            input_schema={
                **base_object,
                "required": ["source"],
                "properties": {
                    "source": {"type": ["string", "array", "object"]},
                    "time": {"type": ["array", "object", "null"]},
                    "output_dir": {"type": ["string", "null"]},
                    "goal": {"type": ["string", "null"]},
                    "include_docs_site": {"type": "boolean"},
                    "include_schemas": {"type": "boolean"},
                },
            },
            output_schema=_artifact_output_schema("handoff"),
            input_example={"source": "my_dataset.npy", "output_dir": "handoff_bundle", "goal": "prepare an agent-ready dataset transfer"},
        ),
        ToolContract(
            tool_id="taskify",
            function_name="tsdataforge.taskify",
            description="Convert a base SeriesDataset into a task-specific TaskDataset only after the asset has been understood.",
            input_schema={
                **base_object,
                "required": ["source", "task"],
                "properties": {
                    "source": {"type": ["string", "array", "object"]},
                    "task": {"type": "string"},
                    "time": {"type": ["array", "object", "null"]},
                    "horizon": {"type": ["integer", "null"]},
                    "window": {"type": ["integer", "null"]},
                    "stride": {"type": ["integer", "null"]},
                },
            },
            output_schema=_artifact_output_schema("taskify"),
            input_example={"source": "my_dataset.npy", "task": "forecasting", "horizon": 24},
        ),
        ToolContract(
            tool_id="demo",
            function_name="tsdataforge.demo",
            description="Generate a built-in demo bundle for the fastest onboarding path or for showcase assets.",
            input_schema={
                **base_object,
                "properties": {
                    "output_dir": {"type": ["string", "null"]},
                    "scenario": {"type": "string"},
                    "n_series": {"type": "integer"},
                    "length": {"type": "integer"},
                    "seed": {"type": "integer"},
                    "include_docs_site": {"type": "boolean"},
                },
            },
            output_schema=_artifact_output_schema("demo"),
            input_example={"output_dir": "demo_bundle", "scenario": "ecg_public", "n_series": 12, "length": 256},
        ),
    )
    return ToolContractCatalog(version=version, tools=tools)


def save_tool_contracts(catalog: ToolContractCatalog | None, output_dir: str | Path) -> ToolContractCatalog:
    cat = catalog or build_tool_contracts()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tool_contracts.json").write_text(json.dumps(cat.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "tool_contracts.md").write_text(cat.to_markdown(), encoding="utf-8")
    return cat


__all__ = ["ToolContract", "ToolContractCatalog", "build_tool_contracts", "save_tool_contracts"]
