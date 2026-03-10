from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..datasets.builder import TaskDataset
from ..datasets.series_dataset import SeriesDataset
from .action_plan import ActionPlanItem
from .cards import ArtifactCard, build_series_dataset_card, build_task_dataset_card
from .context import AgentContextPack, build_dataset_context, build_task_context
from .schemas import build_artifact_schemas, save_artifact_schemas
from .tool_contracts import build_tool_contracts, save_tool_contracts


@dataclass(frozen=True)
class HandoffArtifact:
    """One file or directory produced as part of a handoff bundle."""

    name: str
    path: str
    kind: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "kind": self.kind,
            "description": self.description,
        }


@dataclass(frozen=True)
class HandoffIndex:
    """Compact, explicit first-entry contract for bundle consumers.

    The object can render multiple layers:
    - ``handoff_index_min`` for the smallest possible agent-first entrypoint
    - ``handoff_index`` for a still-compact but more human-readable routing map
    - ``action_plan`` for detailed already_done / recommended / optional steps
    """

    kind: str
    dataset_id: str
    title: str
    summary: str
    wow_sentence: str
    human_open_order: list[str]
    agent_open_order: list[str]
    next_actions: list[str]
    artifact_paths: dict[str, str]
    action_plan: list[dict[str, Any]]
    report_path: str | None = None
    card_path: str | None = None
    context_path: str | None = None
    docs_index: str | None = None
    primary_open_order: list[str] | None = None
    first_non_open_action: str | None = None
    recommended_next_step: str | None = None
    why_recommended: str | None = None
    already_done_actions: list[str] = field(default_factory=list)
    recommended_prompt: str = (
        "Open handoff_index_min.json first. Then follow agent_open_order. Do not open handoff_bundle.json unless a required field is missing."
    )

    def action_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.action_plan:
            status = str(item.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        return counts

    def to_min_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dataset_id": self.dataset_id,
            "title": self.title,
            "wow_sentence": self.wow_sentence,
            "what_this_is": self.summary,
            "agent_entrypoint": "handoff_index_min.json",
            "agent_open_order": list(self.agent_open_order),
            "recommended_next_step": self.recommended_next_step,
            "why_recommended": self.why_recommended,
            "action_plan_path": self.artifact_paths.get("action_plan.json", "action_plan.json"),
            "recommended_prompt": self.recommended_prompt,
        }

    def to_dict(self) -> dict[str, Any]:
        primary_open_order = list(self.primary_open_order or self.human_open_order)
        return {
            "kind": self.kind,
            "dataset_id": self.dataset_id,
            "title": self.title,
            "summary": self.summary,
            "wow_sentence": self.wow_sentence,
            "primary_open_order": primary_open_order,
            "human_open_order": list(self.human_open_order),
            "agent_entrypoint": "handoff_index_min.json",
            "agent_open_order": list(self.agent_open_order),
            "action_plan_path": self.artifact_paths.get("action_plan.json", "action_plan.json"),
            "report_path": self.report_path,
            "card_path": self.card_path,
            "context_path": self.context_path,
            "docs_index": self.docs_index,
            "recommended_next_step": self.recommended_next_step,
            "why_recommended": self.why_recommended,
            "recommended_prompt": self.recommended_prompt,
            "action_status_counts": self.action_status_counts(),
        }

    def to_action_plan_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dataset_id": self.dataset_id,
            "recommended_next_step": self.recommended_next_step,
            "already_done_actions": list(self.already_done_actions),
            "action_status_counts": self.action_status_counts(),
            "action_plan": list(self.action_plan),
        }

    def to_min_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            self.summary,
            "",
            f"> {self.wow_sentence}",
            "",
            "## Agent-first entry",
            "",
            "Open `handoff_index_min.json` first, then follow this order:",
            "",
        ]
        if self.agent_open_order:
            for name in self.agent_open_order:
                target = self.artifact_paths.get(name, name)
                lines.append(f"- `{name}` → `{target}`")
        else:
            lines.append("- No additional artifacts after `handoff_index_min.json`.")
        lines.extend([
            "",
            "## Single recommended next step",
            "",
        ])
        if self.recommended_next_step:
            lines.append(f"- `{self.recommended_next_step}`")
            if self.why_recommended:
                lines.append(f"  - {self.why_recommended}")
            lines.append(f"  - See `action_plan.json` for the full already_done / recommended / optional breakdown.")
        else:
            lines.append("- none")
        lines.extend(["", "## Agent hint", "", self.recommended_prompt, ""])
        return "\n".join(lines).strip() + "\n"

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            self.summary,
            "",
            f"> {self.wow_sentence}",
            "",
            "## Open these first (human)",
            "",
        ]
        if self.human_open_order:
            for name in self.human_open_order:
                target = self.artifact_paths.get(name, name)
                lines.append(f"- `{name}` → `{target}`")
        else:
            lines.append("- No saved artifacts yet.")
        lines.extend([
            "",
            "## Open these first (agent)",
            "",
        ])
        if self.agent_open_order:
            for name in self.agent_open_order:
                target = self.artifact_paths.get(name, name)
                lines.append(f"- `{name}` → `{target}`")
        else:
            lines.append("- No additional artifacts after `handoff_index_min.json`.")
        lines.extend([
            "",
            "## Recommended next step",
            "",
        ])
        if self.recommended_next_step:
            lines.append(f"- `{self.recommended_next_step}`")
            if self.why_recommended:
                lines.append(f"  - {self.why_recommended}")
        else:
            lines.append("- none")
        lines.extend([
            "",
            "## Action plan summary",
            "",
        ])
        counts = self.action_status_counts()
        if counts:
            for key in sorted(counts):
                lines.append(f"- `{key}`: {counts[key]}")
        else:
            lines.append("- none")
        lines.extend(["", "## Agent hint", "", self.recommended_prompt, ""])
        return "\n".join(lines).strip() + "\n"

@dataclass
class DatasetHandoffBundle:
    """Unified package for moving a dataset asset forward."""

    kind: str
    dataset_id: str
    title: str
    summary: str
    context: AgentContextPack
    card: ArtifactCard
    next_actions: list[str]
    manifest: dict[str, Any]
    report: dict[str, Any] | None = None
    artifacts: list[HandoffArtifact] = field(default_factory=list)
    action_plan: list[ActionPlanItem] = field(default_factory=list)
    output_dir: str | None = None
    index: HandoffIndex | None = None

    def to_dict(self) -> dict[str, Any]:
        artifact_names = [item.name for item in self.artifacts]
        open_order = list(self.index.human_open_order) if self.index is not None else []
        payload = {
            "kind": self.kind,
            "dataset_id": self.dataset_id,
            "title": self.title,
            "summary": self.summary,
            "open_order": open_order,
            "next_actions": list(self.next_actions),
            "manifest": dict(self.manifest),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "artifact_names": artifact_names,
            "output_dir": self.output_dir,
            "action_plan": [item.to_dict() for item in self.action_plan],
            "context_preview": {
                "kind": self.context.kind,
                "budget": self.context.budget,
                "estimated_tokens": self.context.estimated_tokens,
                "example_ids": list(self.context.example_ids[:4]),
            },
            "card_preview": {
                "kind": self.card.kind,
                "title": self.card.title,
                "summary": self.card.summary,
            },
        }
        if self.index is not None:
            payload["handoff_index"] = self.index.to_dict()
        if self.report is not None and isinstance(self.report, dict):
            payload["report_preview"] = {
                "title": self.report.get("title"),
                "output_path": self.report.get("output_path"),
                "recommended_tasks": self.report.get("recommended_tasks", [])[:4],
            }
        return payload

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", self.summary, ""]
        if self.index is not None:
            lines.extend([
                "> " + self.index.wow_sentence,
                "",
                "## Open these first (human)",
                "",
            ])
            for name in self.index.human_open_order:
                target = self.index.artifact_paths.get(name, name)
                description = next((item.description for item in self.artifacts if item.name == name), "")
                if description:
                    lines.append(f"- `{name}` — {description} (`{target}`)")
                else:
                    lines.append(f"- `{name}` (`{target}`)")
            lines.extend([
                "",
                "## Open these first (agent)",
                "",
            ])
            for name in self.index.agent_open_order:
                target = self.index.artifact_paths.get(name, name)
                description = next((item.description for item in self.artifacts if item.name == name), "")
                if description:
                    lines.append(f"- `{name}` — {description} (`{target}`)")
                else:
                    lines.append(f"- `{name}` (`{target}`)")
        else:
            lines.extend(["## What to open first", "", "- No saved artifacts yet."])
        lines.extend(["", "## Recommended next actions", ""])
        if self.next_actions:
            lines.extend([f"- `{action}`" for action in self.next_actions])
        else:
            lines.append("- none")
        lines.extend(["", "## Action plan", ""])
        if self.action_plan:
            lines.extend([item.to_markdown() for item in self.action_plan])
        else:
            lines.append("- none")
        lines.extend([
            "",
            "## Why this bundle exists",
            "",
            "This bundle packages the smallest set of artifacts that let a human or an agent understand a time-series dataset asset without reopening notebooks or pasting raw arrays.",
            "",
            "## Artifact inventory",
            "",
        ])
        if self.artifacts:
            for item in self.artifacts:
                lines.append(f"- `{item.name}` ({item.kind}) — {item.description}")
        else:
            lines.append("- none")
        if self.index is not None:
            lines.extend(["", "## Agent hint", "", self.index.recommended_prompt, ""])
        return "\n".join(lines).strip() + "\n"

    def save(self, output_dir: str | Path) -> "DatasetHandoffBundle":
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "handoff_bundle.json").write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        markdown = self.to_markdown()
        (out / "handoff_bundle.md").write_text(markdown, encoding="utf-8")
        if self.index is not None:
            (out / "handoff_index_min.json").write_text(json.dumps(self.index.to_min_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            (out / "handoff_index_min.md").write_text(self.index.to_min_markdown(), encoding="utf-8")
            (out / "handoff_index.json").write_text(json.dumps(self.index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            index_markdown = self.index.to_markdown()
            (out / "handoff_index.md").write_text(index_markdown, encoding="utf-8")
            (out / "action_plan.json").write_text(json.dumps(self.index.to_action_plan_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            (out / "action_plan.md").write_text(_render_action_plan_markdown(self.index), encoding="utf-8")
            (out / "README.md").write_text(self.index.to_min_markdown(), encoding="utf-8")
        else:
            (out / "README.md").write_text(markdown, encoding="utf-8")
        self.output_dir = str(out)
        return self


WOW_SERIES = "Give TSDataForge one raw time-series file and it returns a report, a dataset card, a compact context, and the next actions in about one second."
WOW_TASK = "Give TSDataForge one task dataset and it returns a compact handoff surface that tells the next person or agent what to open and what to do next."


def _render_action_plan_markdown(index: HandoffIndex) -> str:
    lines = [
        f"# Action plan — {index.title}",
        "",
        f"Recommended next step: `{index.recommended_next_step}`" if index.recommended_next_step else "Recommended next step: none",
        "",
    ]
    counts = index.action_status_counts()
    if counts:
        lines.append("## Status counts")
        lines.append("")
        for key in sorted(counts):
            lines.append(f"- `{key}`: {counts[key]}")
        lines.append("")
    lines.extend(["## Actions", ""])
    if index.action_plan:
        for item in index.action_plan:
            lines.append(f"- **{item.get('status', 'recommended')}** · `{item.get('action_id', 'action')}` · {item.get('title', '')}")
            rationale = item.get('rationale')
            if rationale:
                lines.append(f"  - {rationale}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def _artifact(name: str, path: Path, kind: str, description: str) -> HandoffArtifact:
    return HandoffArtifact(name=name, path=str(path), kind=kind, description=description)


def _infer_dataset(values: SeriesDataset | TaskDataset | Any, time: Any | None, *, dataset_id: str | None = None) -> tuple[str, SeriesDataset | TaskDataset]:
    if isinstance(values, SeriesDataset):
        return "series_dataset", values
    if isinstance(values, TaskDataset):
        return "task_dataset", values
    inferred_id = dataset_id or "external_dataset"
    return "series_dataset", SeriesDataset.from_arrays(values, time, dataset_id=inferred_id)


def _task_handoff_summary(task: TaskDataset) -> str:
    y_shape = None if task.y is None else tuple(task.y.shape)
    return (
        f"Task-specialized handoff bundle for `{task.task}`. "
        f"It packages X{tuple(task.X.shape)} and y{y_shape} with a compact context, task card, schema-aware manifest, and recommended next actions."
    )


def _series_open_orders(artifact_paths: dict[str, str]) -> tuple[list[str], list[str], str | None, str | None, str | None]:
    human = [name for name in ("report.html", "dataset_card.md", "dataset_context.json") if name in artifact_paths]
    agent = [name for name in ("dataset_context.json", "dataset_card.md", "report.html") if name in artifact_paths]
    return human, agent, artifact_paths.get("report.html"), artifact_paths.get("dataset_card.md"), artifact_paths.get("dataset_context.json")


def _task_open_orders(artifact_paths: dict[str, str]) -> tuple[list[str], list[str], str | None, str | None, str | None]:
    human = [name for name in ("task_card.md", "task_context.json", "handoff_bundle.md") if name in artifact_paths]
    agent = [name for name in ("task_context.json", "task_card.md") if name in artifact_paths]
    return human, agent, artifact_paths.get("report.html"), artifact_paths.get("task_card.md"), artifact_paths.get("task_context.json")


def _compose_next_actions(kind: str, artifact_paths: dict[str, str], action_plan: list[ActionPlanItem]) -> list[str]:
    if kind == "task_dataset":
        human_open_order, _, _, _, _ = _task_open_orders(artifact_paths)
    else:
        human_open_order, _, _, _, _ = _series_open_orders(artifact_paths)
    open_steps = [f"open:{name}" for name in human_open_order]
    follow_ups = [item.action_id for item in action_plan if item.status != "already_done"]
    return open_steps + follow_ups


def _dataset_quality_flags(context: AgentContextPack) -> dict[str, Any]:
    compact = dict(context.compact)
    top_tags = [item.get("tag", item) if isinstance(item, dict) else item for item in compact.get("top_tags", compact.get("tags", []))]
    dataset_id = str(compact.get("dataset_id") or "")
    goal = str(compact.get("goal") or getattr(context, "goal", None) or "")
    return {
        "recommended_tasks": list(compact.get("recommended_tasks", [])),
        "missing_rate_mean": float(compact.get("missing_rate_mean") or 0.0),
        "dt_cv_mean": float(compact.get("dt_cv_mean") or 0.0),
        "top_tags": [str(item) for item in top_tags],
        "has_trace": bool(compact.get("has_trace")),
        "dataset_id": dataset_id,
        "goal": goal,
    }


def _task_quality_flags(context: AgentContextPack) -> dict[str, Any]:
    compact = dict(context.compact)
    return {
        "task": str(compact.get("task", "task")),
        "aux_keys": list(compact.get("aux_keys", [])),
        "mask_keys": list(compact.get("mask_keys", [])),
    }


def _choose_recommended_task(flags: dict[str, Any]) -> str | None:
    tasks = [str(item) for item in flags.get("recommended_tasks", [])]
    if not tasks:
        return None
    task_set = set(tasks)
    tags = {str(item) for item in flags.get("top_tags", [])}
    dataset_id = str(flags.get("dataset_id", "")).lower()
    goal = str(flags.get("goal", "")).lower()
    text = f"{dataset_id} {goal}"

    if any(key in text for key in ("ecg", "biosignal", "arrhythmia", "clinical", "bedside", "vital")):
        for candidate in ("event_detection", "anomaly_detection", "masked_reconstruction", "forecasting"):
            if candidate in task_set:
                return candidate

    if any(key in text for key in ("factory", "sensor", "drift", "maintenance")):
        for candidate in ("anomaly_detection", "event_detection", "masked_reconstruction", "forecasting"):
            if candidate in task_set:
                return candidate

    if any(key in text for key in ("macro", "inflation", "unemployment", "rates", "regime")):
        for candidate in ("classification", "forecasting", "causal_response"):
            if candidate in task_set:
                return candidate

    if any(key in text for key in ("climate", "co2", "sunspot", "weather", "precip", "temperature")):
        for candidate in ("forecasting", "masked_reconstruction", "change_point_detection", "classification"):
            if candidate in task_set:
                return candidate

    if "missing" in tags:
        for candidate in ("masked_reconstruction", "anomaly_detection", "forecasting"):
            if candidate in task_set:
                return candidate

    if "bursty" in tags:
        for candidate in ("event_detection", "anomaly_detection", "forecasting"):
            if candidate in task_set:
                return candidate

    if any(tag in tags for tag in ("regime", "piecewise", "chaotic")):
        for candidate in ("classification", "forecasting", "causal_response"):
            if candidate in task_set:
                return candidate

    return tasks[0]


def _series_action_plan(
    context: AgentContextPack,
    *,
    include_report: bool,
    include_schemas: bool,
    include_docs_site: bool,
) -> list[ActionPlanItem]:
    flags = _dataset_quality_flags(context)
    tasks = [str(item) for item in flags.get("recommended_tasks", [])]
    top_tags = ", ".join(flags.get("top_tags", [])[:3]) or "the dominant structure tags"
    actions: list[ActionPlanItem] = []
    if include_report:
        actions.append(
            ActionPlanItem(
                action_id="done:save_report",
                title="Saved the outcome-first HTML report",
                kind="save",
                status="already_done",
                rationale="The report is already present, so the next step should interpret it rather than regenerate it.",
                target="report.html",
                related_artifacts=("report.html",),
            )
        )
    actions.extend(
        [
            ActionPlanItem(
                action_id="done:save_dataset_card",
                title="Saved the dataset card",
                kind="save",
                status="already_done",
                rationale="The card is already present and should be reused for teammate handoff instead of rewritten in chat.",
                target="dataset_card.md",
                related_artifacts=("dataset_card.md", "dataset_card.json"),
            ),
            ActionPlanItem(
                action_id="done:save_dataset_context",
                title="Saved the compact dataset context",
                kind="save",
                status="already_done",
                rationale="The compact context is already available and should be the machine-readable summary for prompts and agents.",
                target="dataset_context.json",
                related_artifacts=("dataset_context.json", "dataset_context.md"),
            ),
            ActionPlanItem(
                action_id="done:save_handoff_index",
                title="Saved the first-entry handoff index",
                kind="save",
                status="already_done",
                rationale="The minimal index is already present, so agents can start from a tiny contract instead of a large JSON inventory.",
                target="handoff_index_min.json",
                related_artifacts=("handoff_index_min.json", "handoff_index_min.md", "handoff_index.json", "handoff_index.md"),
            ),
        ]
    )
    if include_schemas:
        actions.append(
            ActionPlanItem(
                action_id="done:save_schemas",
                title="Saved artifact schemas",
                kind="save",
                status="already_done",
                rationale="Schema contracts are already bundled, so external tools can validate artifacts without reading the Python source.",
                target="schemas/",
                related_artifacts=("schemas/",),
            )
        )
    if include_docs_site:
        actions.append(
            ActionPlanItem(
                action_id="done:render_docs_site",
                title="Rendered the offline docs site",
                kind="save",
                status="already_done",
                rationale="A local docs site is already bundled for workshops, screenshots, and offline review.",
                target="docs/index.html",
                related_artifacts=("docs/",),
            )
        )

    recommended_task = _choose_recommended_task(flags)
    if flags["missing_rate_mean"] >= 0.05:
        actions.append(
            ActionPlanItem(
                action_id="review:missingness_strategy",
                title="Review missingness and monitoring gaps before modeling",
                kind="review",
                status="recommended",
                rationale=f"Mean missing rate is about {flags['missing_rate_mean']:.3f}; decide masking, interpolation, or robust task routing before training.",
                trigger="recommended when missingness is non-trivial",
                command_hint="taskify(base, task='masked_reconstruction', ...) or document the imputation policy",
                related_artifacts=("report.html", "dataset_context.json"),
            )
        )
    elif flags["dt_cv_mean"] >= 0.10:
        actions.append(
            ActionPlanItem(
                action_id="review:irregular_sampling_strategy",
                title="Review irregular sampling before committing to a task",
                kind="review",
                status="recommended",
                rationale=f"Sampling irregularity (dt_cv_mean≈{flags['dt_cv_mean']:.3f}) is large enough that the observation mechanism should be handled explicitly.",
                trigger="recommended when irregular sampling dominates",
                command_hint="document the observation policy or compare regular-vs-irregular task views",
                related_artifacts=("report.html", "dataset_context.json"),
            )
        )
    elif recommended_task is not None:
        actions.append(
            ActionPlanItem(
                action_id=f"run:taskify:{recommended_task}",
                title=f"Taskify into `{recommended_task}`",
                kind="run",
                status="recommended",
                rationale=f"This is the top suggested task because the bundle is dominated by {top_tags}.",
                trigger="recommended after the report is reviewed",
                command_hint=f"taskify(base, task='{recommended_task}', ...)",
                related_artifacts=("report.html", "dataset_context.json", "dataset_card.md"),
            )
        )
    else:
        actions.append(
            ActionPlanItem(
                action_id="review:report_findings",
                title="Review the report findings and choose a downstream task",
                kind="review",
                status="recommended",
                rationale="No single downstream task dominates, so the most useful next step is to interpret the report before taskifying.",
                trigger="recommended when routing is ambiguous",
                command_hint="open report.html and inspect the linked docs/examples/API suggestions",
                related_artifacts=("report.html",),
            )
        )

    for task in tasks[:3]:
        action_id = f"run:taskify:{task}"
        if any(item.action_id == action_id for item in actions):
            continue
        actions.append(
            ActionPlanItem(
                action_id=action_id,
                title=f"Taskify into `{task}`",
                kind="run",
                status="optional",
                rationale=f"Alternative task route suggested by the compact context for the same asset ({top_tags}).",
                trigger="use when the downstream objective is fixed",
                command_hint=f"taskify(base, task='{task}', ...)",
                related_artifacts=("dataset_context.json",),
            )
        )
    actions.append(
        ActionPlanItem(
            action_id="compare:dataset_versions",
            title="Compare this asset against the previous dataset version",
            kind="compare",
            status="optional",
            rationale="Version-to-version drift is a common next move once a first handoff bundle exists.",
            trigger="use when you have v1 and v2 of the same asset",
            command_hint="see examples/compare_two_dataset_versions.py",
            related_artifacts=("dataset_card.md", "dataset_context.json"),
        )
    )
    return actions


def _task_action_plan(context: AgentContextPack, *, include_schemas: bool) -> list[ActionPlanItem]:
    flags = _task_quality_flags(context)
    task = flags.get("task", "task")
    actions: list[ActionPlanItem] = [
        ActionPlanItem(
            action_id="done:save_task_card",
            title="Saved the task card",
            kind="save",
            status="already_done",
            rationale="The task card is already present and should be the human-readable transfer summary.",
            target="task_card.md",
            related_artifacts=("task_card.md", "task_card.json"),
        ),
        ActionPlanItem(
            action_id="done:save_task_context",
            title="Saved the compact task context",
            kind="save",
            status="already_done",
            rationale="The task context is already present and should be the low-token machine interface.",
            target="task_context.json",
            related_artifacts=("task_context.json", "task_context.md"),
        ),
        ActionPlanItem(
            action_id="done:save_handoff_index",
            title="Saved the first-entry handoff index",
            kind="save",
            status="already_done",
            rationale="The minimal index is already present and is the intended first stop for an agent.",
            target="handoff_index_min.json",
            related_artifacts=("handoff_index_min.json", "handoff_index_min.md", "handoff_index.json", "handoff_index.md"),
        ),
    ]
    if include_schemas:
        actions.append(
            ActionPlanItem(
                action_id="done:save_schemas",
                title="Saved artifact schemas",
                kind="save",
                status="already_done",
                rationale="Schema contracts are already bundled for downstream tooling and validation.",
                target="schemas/",
                related_artifacts=("schemas/",),
            )
        )
    actions.append(
        ActionPlanItem(
            action_id="review:schema",
            title="Inspect the task schema before modeling",
            kind="review",
            status="recommended",
            rationale=f"`{task}` is already taskified, so the most useful next step is to verify X/y semantics, masks, and aux keys before training.",
            trigger="always",
            command_hint="open task_context.json and task_card.md before loading raw arrays",
            related_artifacts=("task_context.json", "task_card.md"),
        )
    )
    actions.append(
        ActionPlanItem(
            action_id="train:baseline",
            title="Train or evaluate a baseline on the task dataset",
            kind="run",
            status="optional",
            rationale="Once the schema is understood, this task artifact is ready for model code or agent orchestration.",
            trigger="after schema review",
            command_hint="see task-specific examples in examples/",
            related_artifacts=("task_context.json",),
        )
    )
    return actions


def _build_index(bundle: DatasetHandoffBundle, *, docs_index: str | None = None) -> HandoffIndex:
    artifact_paths = {item.name: item.path for item in bundle.artifacts}
    if bundle.kind == "task_dataset":
        human_open_order, agent_open_order, report_path, card_path, context_path = _task_open_orders(artifact_paths)
        wow_sentence = WOW_TASK
    else:
        human_open_order, agent_open_order, report_path, card_path, context_path = _series_open_orders(artifact_paths)
        wow_sentence = WOW_SERIES
    first_non_open = next((item.action_id for item in bundle.action_plan if item.status == "recommended"), None)
    recommended = next((item for item in bundle.action_plan if item.status == "recommended"), None)
    already_done = [item.action_id for item in bundle.action_plan if item.status == "already_done"]
    return HandoffIndex(
        kind=bundle.kind,
        dataset_id=bundle.dataset_id,
        title=bundle.title,
        summary=bundle.summary,
        wow_sentence=wow_sentence,
        human_open_order=human_open_order,
        agent_open_order=agent_open_order,
        primary_open_order=list(human_open_order),
        next_actions=list(bundle.next_actions),
        action_plan=[item.to_dict() for item in bundle.action_plan],
        artifact_paths=artifact_paths,
        report_path=report_path,
        card_path=card_path,
        context_path=context_path,
        docs_index=docs_index,
        first_non_open_action=first_non_open,
        recommended_next_step=recommended.action_id if recommended is not None else None,
        why_recommended=recommended.rationale if recommended is not None else None,
        already_done_actions=already_done,
        recommended_prompt=(
            "Open handoff_index_min.json first. Then follow agent_open_order. Summarize dataset quality, main risks, and the single most useful next step. Open action_plan.json only if you need more detail. Do not open handoff_bundle.json unless a required field is missing."
        ),
    )


def render_handoff_index_markdown(index: HandoffIndex) -> str:
    return index.to_markdown()


def render_dataset_handoff_markdown(bundle: DatasetHandoffBundle) -> str:
    return bundle.to_markdown()


def save_dataset_handoff_bundle(bundle: DatasetHandoffBundle, output_dir: str | Path) -> DatasetHandoffBundle:
    return bundle.save(output_dir)


def build_dataset_handoff_bundle(
    values: SeriesDataset | TaskDataset | Any,
    time: Any | None = None,
    *,
    output_dir: str | Path | None = None,
    goal: str | None = None,
    title: str | None = None,
    include_report: bool = True,
    include_docs_site: bool = False,
    include_source_asset: bool = True,
    include_schemas: bool = True,
    max_series: int | None = 200,
    seed: int = 0,
    dataset_id: str | None = None,
    docs_title: str = "TSDataForge Docs",
    docs_base_url: str | None = None,
) -> DatasetHandoffBundle:
    """Build the shortest happy-path asset bundle for a dataset."""

    kind, obj = _infer_dataset(values, time, dataset_id=dataset_id)
    artifacts: list[HandoffArtifact] = []
    report_payload: dict[str, Any] | None = None
    docs_index: str | None = None

    if kind == "task_dataset":
        task = obj  # type: ignore[assignment]
        assert isinstance(task, TaskDataset)
        context = build_task_context(task, budget="small", goal=goal)
        card = build_task_dataset_card(task, goal=goal)
        data_id = f"task:{task.task}"
        summary = _task_handoff_summary(task)
        manifest = {
            "kind": kind,
            "dataset_id": data_id,
            "task": task.task,
            "schema": task.schema or {},
            "n_samples": int(len(task.X)),
            "goal": goal,
        }
    else:
        dataset = obj  # type: ignore[assignment]
        assert isinstance(dataset, SeriesDataset)
        context = build_dataset_context(dataset, budget="small", goal=goal, max_series=max_series)
        card = build_series_dataset_card(dataset, goal=goal)
        data_id = dataset.dataset_id
        summary = (
            f"Time-series dataset report + handoff bundle for `{data_id}`. "
            f"It packages an EDA-first explanation, compact context, dataset card, a tiny handoff index, and explicit next actions so the asset can move cleanly between people, notebooks, and agents."
        )
        manifest = {
            "kind": kind,
            "dataset_id": data_id,
            "n_series": int(len(dataset.series)),
            "channel_names": dataset.channel_names,
            "goal": goal,
        }

    bundle = DatasetHandoffBundle(
        kind=kind,
        dataset_id=data_id,
        title=title or (f"TSDataForge Handoff Bundle — {data_id}"),
        summary=summary,
        context=context,
        card=card,
        next_actions=[],
        manifest=manifest,
        artifacts=artifacts,
        action_plan=[],
    )

    if output_dir is None:
        bundle.action_plan = _task_action_plan(context, include_schemas=include_schemas) if kind == "task_dataset" else _series_action_plan(context, include_report=include_report, include_schemas=include_schemas, include_docs_site=include_docs_site)
        bundle.next_actions = [item.action_id for item in bundle.action_plan if item.status != "already_done"]
        bundle.index = _build_index(bundle)
        bundle.next_actions = _compose_next_actions(bundle.kind, {item.name: item.path for item in bundle.artifacts}, bundle.action_plan)
        bundle.index = _build_index(bundle)
        return bundle

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if include_source_asset:
        if kind == "task_dataset":
            assert isinstance(obj, TaskDataset)
            obj.save(out / "asset", include_card=False, include_agent_context=False, include_handoff_bundle=False)
            artifacts.append(_artifact("asset/", out / "asset", "task_dataset", "TaskDataset arrays and manifest."))
        else:
            assert isinstance(obj, SeriesDataset)
            obj.save(out / "asset", include_card=False, include_agent_context=False, include_handoff_bundle=False)
            artifacts.append(_artifact("asset/", out / "asset", "series_dataset", "Base SeriesDataset arrays and manifest."))

    if kind == "task_dataset":
        context_json = out / "task_context.json"
        context_md = out / "task_context.md"
        card_json = out / "task_card.json"
        card_md = out / "task_card.md"
    else:
        context_json = out / "dataset_context.json"
        context_md = out / "dataset_context.md"
        card_json = out / "dataset_card.json"
        card_md = out / "dataset_card.md"

    context_json.write_text(json.dumps(context.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    context_md.write_text(context.to_markdown(), encoding="utf-8")
    artifacts.append(_artifact(context_json.name, context_json, "context", "Compact, low-token context for human or agent handoff."))
    artifacts.append(_artifact(context_md.name, context_md, "context_markdown", "Human-readable context version."))

    card_json.write_text(json.dumps(card.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    card_md.write_text(card.to_markdown(), encoding="utf-8")
    artifacts.append(_artifact(card_json.name, card_json, "card", "Machine-readable dataset or task card."))
    artifacts.append(_artifact(card_md.name, card_md, "card_markdown", "Human-readable dataset or task card."))

    if include_schemas:
        schema_dir = out / "schemas"
        schema_catalog = save_artifact_schemas(build_artifact_schemas(version="0.3.7"), schema_dir)
        tool_catalog = save_tool_contracts(build_tool_contracts(version="0.3.7"), schema_dir)
        manifest["schemas"] = [item.schema_id for item in schema_catalog.schemas]
        manifest["tool_contracts"] = [item.tool_id for item in tool_catalog.tools]
        artifacts.append(_artifact("schemas/", schema_dir, "schema_catalog", "JSON Schema contracts for context, cards, handoff indices, action plans, and handoff bundles."))
        artifacts.append(_artifact("schemas/tool_contracts.json", schema_dir / "tool_contracts.json", "tool_contracts", "Structured tool-calling contracts for the five public entry points."))

    if include_docs_site:
        from .site import generate_docs_site

        site = generate_docs_site(out / "docs", title=docs_title)
        docs_index = str(out / "docs" / "index.html")
        manifest["docs_index"] = docs_index
        manifest["docs_site"] = site.to_dict()
        artifacts.append(_artifact("docs/", out / "docs", "docs_site", "Offline docs site linked from the report."))
        docs_base_url = "docs/"
    elif docs_base_url is None:
        docs_base_url = None

    if include_report and kind == "series_dataset":
        from ..report.eda import generate_dataset_eda_report

        report = generate_dataset_eda_report(
            obj.values_list(),
            obj.time_list(),
            title=f"TSDataForge Dataset Report — {data_id}",
            output_path=out / "report.html",
            max_series=max_series,
            seed=seed,
            docs_base_url=docs_base_url,
            include_linked_resources=True,
        )
        report_payload = report.to_dict()
        manifest["report"] = str(out / "report.html")
        artifacts.append(_artifact("report.html", out / "report.html", "eda_report", "Outcome-first dataset EDA report."))
        hub_json = out / "report_resource_hub.json"
        hub_md = out / "report_resource_hub.md"
        if report.resource_hub is not None:
            hub_json.write_text(json.dumps(report.resource_hub.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            from .eda_linking import render_eda_resource_hub_markdown

            hub_md.write_text(render_eda_resource_hub_markdown(report.resource_hub), encoding="utf-8")
            artifacts.append(_artifact(hub_json.name, hub_json, "resource_hub", "Machine-readable routing map from report findings to docs/examples/API/FAQ."))
            artifacts.append(_artifact(hub_md.name, hub_md, "resource_hub_markdown", "Human-readable routing map from report findings to docs/examples/API/FAQ."))
    elif include_report and kind == "task_dataset":
        manifest["report"] = None
        manifest["report_note"] = "TaskDataset bundles skip dataset-level EDA by default; inspect the task card and schema first."

    bundle.report = report_payload
    bundle.artifacts = artifacts
    bundle.action_plan = _task_action_plan(context, include_schemas=include_schemas) if kind == "task_dataset" else _series_action_plan(context, include_report=include_report, include_schemas=include_schemas, include_docs_site=include_docs_site)
    bundle.next_actions = _compose_next_actions(kind, {item.name: item.path for item in artifacts}, bundle.action_plan)
    bundle.manifest = dict(manifest)
    bundle.index = _build_index(bundle, docs_index=docs_index)
    bundle.artifacts.extend([
        _artifact("handoff_index_min.json", out / "handoff_index_min.json", "handoff_index_min", "Smallest agent-first routing contract for the bundle."),
        _artifact("handoff_index_min.md", out / "handoff_index_min.md", "handoff_index_min_markdown", "Human-readable version of the minimal handoff index."),
        _artifact("handoff_index.json", out / "handoff_index.json", "handoff_index", "Compact routing map that expands the minimal index for human and agent open order."),
        _artifact("handoff_index.md", out / "handoff_index.md", "handoff_index_markdown", "Human-readable routing map for the handoff bundle."),
        _artifact("action_plan.json", out / "action_plan.json", "action_plan", "Detailed already_done / recommended / optional steps for the bundle."),
        _artifact("action_plan.md", out / "action_plan.md", "action_plan_markdown", "Human-readable action plan for the bundle."),
        _artifact("handoff_bundle.json", out / "handoff_bundle.json", "handoff_bundle", "Inventory of bundle artifacts and previews of the main handoff surfaces."),
        _artifact("handoff_bundle.md", out / "handoff_bundle.md", "handoff_bundle_markdown", "Human-readable inventory for the bundle."),
    ])
    bundle.save(out)
    return bundle


__all__ = [
    "ActionPlanItem",
    "HandoffArtifact",
    "HandoffIndex",
    "DatasetHandoffBundle",
    "build_dataset_handoff_bundle",
    "render_handoff_index_markdown",
    "render_dataset_handoff_markdown",
    "save_dataset_handoff_bundle",
]
