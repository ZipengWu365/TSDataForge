from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..datasets.builder import TaskDataset
from ..datasets.series_dataset import SeriesDataset
from .context import build_dataset_context, build_task_context


@dataclass
class ArtifactCard:
    kind: str
    title: str
    summary: str
    intended_use: list[str] = field(default_factory=list)
    quickstart: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    key_fields: dict[str, Any] = field(default_factory=dict)
    example_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        sections = [f"# {self.title}", "", self.summary, ""]
        sections.append("## Intended use")
        sections.extend([f"- {x}" for x in self.intended_use] or ["- none specified"])
        sections.append("")
        sections.append("## Quickstart")
        sections.extend([f"- {x}" for x in self.quickstart] or ["- none specified"])
        sections.append("")
        sections.append("## Caveats")
        sections.extend([f"- {x}" for x in self.caveats] or ["- none specified"])
        sections.append("")
        sections.append("## Key fields")
        sections.append("```json")
        sections.append(json.dumps(self.key_fields, ensure_ascii=False, indent=2))
        sections.append("```")
        sections.append("")
        sections.append("## Suggested examples")
        sections.append(", ".join(self.example_ids) if self.example_ids else "none")
        sections.append("")
        return "\n".join(sections)



def build_series_dataset_card(dataset: SeriesDataset, *, goal: str | None = None) -> ArtifactCard:
    pack = build_dataset_context(dataset, budget="small", goal=goal)
    compact = dict(pack.compact)
    title = f"SeriesDataset card: {compact.get('dataset_id', dataset.dataset_id)}"
    summary = (
        f"Base dataset with {compact.get('n_series')} series. "
        f"Top tags: {', '.join([item['tag'] for item in compact.get('top_tags', [])]) or 'none'}."
    )
    intended = [
        "Use as the raw substrate for EDA, structure description, and multi-task derivation.",
        "Taskify into forecasting / classification / causal / control tasks instead of maintaining separate dataset scripts.",
    ]
    if compact.get("has_trace"):
        intended.append("Trace is present, so change/event/intervention/counterfactual tasks may be derivable without relabeling.")
    quickstart = [
        "build_dataset_handoff_bundle(base, output_dir='bundle') for the shortest report + handoff path.",
        "describe_dataset(...) or generate_dataset_eda_report(...) when you need standalone analysis.",
        "base.taskify(task='forecasting', ...) for prediction.",
        "build_dataset_context(base) when you only need a compact prompt surface.",
    ]
    caveats = [
        "Base datasets do not imply one canonical task; read the recommended_tasks list first.",
        "Real datasets created from arrays will typically not have causal or changepoint truth unless you attach it yourself.",
    ]
    return ArtifactCard(
        kind="series_dataset",
        title=title,
        summary=summary,
        intended_use=intended,
        quickstart=quickstart,
        caveats=caveats,
        key_fields=compact,
        example_ids=pack.example_ids,
    )



def build_task_dataset_card(dataset: TaskDataset, *, goal: str | None = None) -> ArtifactCard:
    pack = build_task_context(dataset, budget="small", goal=goal)
    compact = dict(pack.compact)
    task = dataset.task
    title = f"TaskDataset card: {task}"
    summary = (
        f"Task-specialized dataset for `{task}` with X{tuple(compact.get('X_shape', []))} "
        f"and y{tuple(compact.get('y_shape', [])) if compact.get('y_shape') is not None else 'None'}."
    )
    intended = {
        "forecasting": ["Use for horizon-based prediction baselines and sequence models."],
        "classification": ["Use for structure classification or representation learning evaluation."],
        "system_identification": ["Use for learning input-output dynamics from past [u, y] to future y."],
        "causal_response": ["Use for treatment/outcome response forecasting under temporal confounding."],
        "counterfactual_response": ["Use for factual vs counterfactual evaluation or policy comparison."],
        "policy_value_estimation": ["Use for return prediction and policy evaluation."],
    }.get(task, ["Use the explicit schema to wire this task into training or evaluation pipelines."])
    quickstart = [
        "Read dataset.schema before writing model code or prompts.",
        "build_dataset_handoff_bundle(task_ds, output_dir='bundle') when the task artifact needs to move between people or agents.",
        "Persist the card next to the data so agents can understand the artifact without reopening notebooks.",
        "Prefer compact context packs over pasting raw arrays or long README text into prompts.",
    ]
    caveats = [
        "Task datasets are downstream views. If you need another task, regenerate from the base dataset instead of mutating X/y in-place.",
        "Prefer bundling task card + task context before handing the asset to another person or agent.",
        "Some tasks only make sense when trace carries the necessary truth (for example adjacency or ITE).",
    ]
    return ArtifactCard(
        kind="task_dataset",
        title=title,
        summary=summary,
        intended_use=intended,
        quickstart=quickstart,
        caveats=caveats,
        key_fields=compact,
        example_ids=pack.example_ids,
    )



def save_card(card: ArtifactCard, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(card.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if path.suffix.lower() in {".md", ".markdown"}:
        path.write_text(card.to_markdown(), encoding="utf-8")
        return
    raise ValueError("Unsupported card extension. Use .json or .md")
