from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .context import AgentContextPack


@dataclass(frozen=True)
class DecisionFact:
    fact_id: str
    label: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionRisk:
    risk_id: str
    title: str
    severity: str
    rationale: str
    affected_tasks: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateTaskDecision:
    task: str
    score: float
    rationale: str
    blocked_by: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 3)
        return payload


@dataclass(frozen=True)
class RecommendedNextStep:
    action_id: str
    title: str
    kind: str
    rationale: str
    confidence: float
    command_hint: str | None = None
    why_not: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 3)
        return payload


@dataclass(frozen=True)
class DecisionRecord:
    kind: str
    dataset_id: str
    goal: str | None
    summary: str
    facts: tuple[DecisionFact, ...] = field(default_factory=tuple)
    risks: tuple[DecisionRisk, ...] = field(default_factory=tuple)
    blockers: tuple[DecisionRisk, ...] = field(default_factory=tuple)
    candidate_tasks: tuple[CandidateTaskDecision, ...] = field(default_factory=tuple)
    recommended_next_step: RecommendedNextStep | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dataset_id": self.dataset_id,
            "goal": self.goal,
            "summary": self.summary,
            "facts": [item.to_dict() for item in self.facts],
            "risks": [item.to_dict() for item in self.risks],
            "blockers": [item.to_dict() for item in self.blockers],
            "candidate_tasks": [item.to_dict() for item in self.candidate_tasks],
            "recommended_next_step": None if self.recommended_next_step is None else self.recommended_next_step.to_dict(),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Decision record: {self.dataset_id}",
            "",
            self.summary,
            "",
        ]
        if self.goal:
            lines.extend(["## Goal", "", f"- {self.goal}", ""])
        lines.extend(["## Facts", ""])
        if self.facts:
            for item in self.facts:
                lines.append(f"- **{item.label}**: {item.value}")
        else:
            lines.append("- none")
        lines.extend(["", "## Main risks", ""])
        if self.risks:
            for item in self.risks:
                tasks = f" Affects: {', '.join(item.affected_tasks)}." if item.affected_tasks else ""
                lines.append(f"- **{item.severity}** `{item.risk_id}`: {item.title}. {item.rationale}{tasks}")
        else:
            lines.append("- none")
        lines.extend(["", "## Blockers", ""])
        if self.blockers:
            for item in self.blockers:
                tasks = f" Affects: {', '.join(item.affected_tasks)}." if item.affected_tasks else ""
                lines.append(f"- **{item.severity}** `{item.risk_id}`: {item.title}. {item.rationale}{tasks}")
        else:
            lines.append("- none")
        lines.extend(["", "## Candidate tasks", ""])
        if self.candidate_tasks:
            for item in self.candidate_tasks:
                blocked = f" Blocked by: {', '.join(item.blocked_by)}." if item.blocked_by else ""
                lines.append(f"- `{item.task}` score={item.score:.3f}. {item.rationale}{blocked}")
        else:
            lines.append("- none")
        lines.extend(["", "## Recommended next step", ""])
        if self.recommended_next_step is not None:
            lines.append(
                f"- `{self.recommended_next_step.action_id}` ({self.recommended_next_step.kind}, confidence={self.recommended_next_step.confidence:.3f})"
            )
            lines.append(f"  - {self.recommended_next_step.title}")
            lines.append(f"  - {self.recommended_next_step.rationale}")
            if self.recommended_next_step.command_hint:
                lines.append(f"  - Hint: `{self.recommended_next_step.command_hint}`")
            for item in self.recommended_next_step.why_not:
                lines.append(f"  - Why not: {item}")
        else:
            lines.append("- none")
        lines.append("")
        return "\n".join(lines)

    def top_risk_titles(self, limit: int = 3) -> list[str]:
        return [item.title for item in self.risks[:limit]]

    def top_candidate_task_names(self, limit: int = 3) -> list[str]:
        return [item.task for item in self.candidate_tasks[:limit]]


def _goal_text(flags: dict[str, Any]) -> str:
    dataset_id = str(flags.get("dataset_id", "")).lower()
    goal = str(flags.get("goal", "")).lower()
    return f"{dataset_id} {goal}".strip()


def _task_hint(task: str) -> str:
    return {
        "forecasting": "taskify(base, task='forecasting', ...)",
        "masked_reconstruction": "taskify(base, task='masked_reconstruction', ...) or document the imputation policy",
        "anomaly_detection": "taskify(base, task='anomaly_detection', ...)",
        "event_detection": "taskify(base, task='event_detection', ...)",
        "change_point_detection": "taskify(base, task='change_point_detection', ...)",
        "classification": "taskify(base, task='classification', ...)",
        "system_identification": "taskify(base, task='system_identification', ...)",
        "causal_response": "taskify(base, task='causal_response', ...)",
        "counterfactual_response": "taskify(base, task='counterfactual_response', ...)",
    }.get(task, f"taskify(base, task='{task}', ...)")


def _score_candidate_task(task: str, *, flags: dict[str, Any], rank: int) -> tuple[float, str]:
    tags = set(str(item) for item in flags.get("top_tags", []))
    text = _goal_text(flags)
    score = max(0.18, 0.56 - 0.08 * rank)
    reasons: list[str] = []

    if task == "forecasting":
        if {"trend", "seasonal", "random_walk_like", "ar1_like"} & tags:
            score += 0.22
            reasons.append("trend/seasonal structure is visible")
        if any(key in text for key in ("climate", "macro", "inflation", "co2", "weather", "sales")):
            score += 0.08
            reasons.append("the goal reads like a forecasting-style use case")
        if float(flags.get("missing_rate_mean", 0.0)) >= 0.05:
            score -= 0.08
            reasons.append("missingness weakens forecasting readiness")
        if float(flags.get("dt_cv_mean", 0.0)) >= 0.10:
            score -= 0.06
            reasons.append("irregular sampling should be handled first")
    elif task == "masked_reconstruction":
        if float(flags.get("missing_rate_mean", 0.0)) >= 0.05:
            score += 0.24
            reasons.append("missingness is already non-trivial")
        if float(flags.get("dt_cv_mean", 0.0)) >= 0.10:
            score += 0.12
            reasons.append("observation issues are part of the problem")
        if "missing" in tags or "irregular_sampling" in tags:
            score += 0.10
            reasons.append("structure tags already point to observation repair")
    elif task in {"anomaly_detection", "event_detection"}:
        if {"bursty", "heavy_tail"} & tags:
            score += 0.22
            reasons.append("bursty or heavy-tail behavior is present")
        if any(key in text for key in ("sensor", "drift", "monitor", "ecg", "vital", "factory", "maintenance")):
            score += 0.08
            reasons.append("the goal looks event or anomaly oriented")
        if task == "event_detection" and "bursty" in tags:
            score += 0.05
    elif task in {"classification", "system_identification", "causal_response"}:
        if {"multivariate", "coupled"} & tags:
            score += 0.18
            reasons.append("multi-channel structure supports richer downstream tasks")
        if any(key in text for key in ("regime", "class", "state", "policy", "control", "causal")):
            score += 0.08
            reasons.append("the goal text points to a structured downstream task")
    elif task == "change_point_detection":
        if {"regime", "piecewise", "bursty"} & tags:
            score += 0.20
            reasons.append("state changes or bursts suggest breakpoints")
        if any(key in text for key in ("change", "regime", "drift", "switch")):
            score += 0.08
            reasons.append("the goal text points to changepoints")
    elif task in {"counterfactual_response", "causal_discovery", "causal_ite"}:
        if bool(flags.get("has_trace")):
            score += 0.18
            reasons.append("trace is present")
        if any(key in text for key in ("counterfactual", "intervention", "treatment", "causal")):
            score += 0.08
            reasons.append("the goal explicitly asks for causal reasoning")

    score = min(0.99, max(0.05, score))
    rationale = "; ".join(reasons) if reasons else "Suggested by the compact context and route map."
    return score, rationale


def _dataset_risks(flags: dict[str, Any]) -> tuple[list[DecisionRisk], list[DecisionRisk]]:
    risks: list[DecisionRisk] = []
    blockers: list[DecisionRisk] = []
    missing = float(flags.get("missing_rate_mean", 0.0))
    irregular = float(flags.get("dt_cv_mean", 0.0))
    has_trace = bool(flags.get("has_trace"))

    if missing >= 0.10:
        risks.append(
            DecisionRisk(
                risk_id="missingness_high",
                title="Missingness is high enough to distort downstream tasks",
                severity="high",
                rationale=f"Mean missing rate is about {missing:.2%}; repair or observation modeling should come before task commitment.",
                affected_tasks=("forecasting", "classification", "causal_response"),
            )
        )
    elif missing >= 0.05:
        risks.append(
            DecisionRisk(
                risk_id="missingness_medium",
                title="Missingness should be reviewed before modeling",
                severity="medium",
                rationale=f"Mean missing rate is about {missing:.2%}; document masking or interpolation before training.",
                affected_tasks=("forecasting", "classification"),
            )
        )

    if irregular >= 0.20:
        risks.append(
            DecisionRisk(
                risk_id="irregular_sampling_high",
                title="Sampling is highly irregular",
                severity="high",
                rationale=f"dt_cv_mean is about {irregular:.3f}; direct forecasting baselines may hide an observation problem.",
                affected_tasks=("forecasting", "change_point_detection"),
            )
        )
    elif irregular >= 0.10:
        risks.append(
            DecisionRisk(
                risk_id="irregular_sampling_medium",
                title="Sampling irregularity should be handled explicitly",
                severity="medium",
                rationale=f"dt_cv_mean is about {irregular:.3f}; compare regular-vs-irregular views before taskifying.",
                affected_tasks=("forecasting",),
            )
        )

    if not has_trace:
        blockers.append(
            DecisionRisk(
                risk_id="trace_not_available",
                title="Trace-only tasks are not fully grounded",
                severity="high",
                rationale="This asset has no built-in trace or intervention truth, so counterfactual, ITE, and trace-grounded discovery tasks need extra labels or synthetic support.",
                affected_tasks=("counterfactual_response", "causal_ite", "causal_discovery"),
            )
        )

    return risks, blockers


def build_dataset_decision_record(context: AgentContextPack) -> DecisionRecord:
    compact = dict(context.compact)
    flags = {
        "recommended_tasks": list(compact.get("recommended_tasks", [])),
        "missing_rate_mean": float(compact.get("missing_rate_mean") or 0.0),
        "dt_cv_mean": float(compact.get("dt_cv_mean") or 0.0),
        "top_tags": [item.get("tag", item) if isinstance(item, dict) else item for item in compact.get("top_tags", compact.get("tags", []))],
        "has_trace": bool(compact.get("has_trace")),
        "dataset_id": str(compact.get("dataset_id") or "dataset"),
        "goal": str(compact.get("goal") or ""),
    }
    facts = (
        DecisionFact("n_series", "Series count", str(compact.get("n_series", "n/a"))),
        DecisionFact("length_median", "Median length", str(compact.get("length_median", "n/a"))),
        DecisionFact("channels_median", "Median channels", str(compact.get("channels_median", "n/a"))),
        DecisionFact("top_tags", "Top tags", ", ".join(str(item) for item in flags["top_tags"][:4]) or "none"),
        DecisionFact("missing_rate_mean", "Mean missing rate", f"{flags['missing_rate_mean']:.2%}"),
        DecisionFact("dt_cv_mean", "Mean dt CV", f"{flags['dt_cv_mean']:.3f}"),
        DecisionFact("has_trace", "Trace attached", "yes" if flags["has_trace"] else "no"),
    )
    risks, blockers = _dataset_risks(flags)
    blocked_by_task = {
        task: tuple(item.risk_id for item in blockers if task in item.affected_tasks)
        for task in flags["recommended_tasks"]
    }

    scored: list[CandidateTaskDecision] = []
    for idx, task in enumerate(flags["recommended_tasks"]):
        score, rationale = _score_candidate_task(str(task), flags=flags, rank=idx)
        scored.append(
            CandidateTaskDecision(
                task=str(task),
                score=score,
                rationale=rationale,
                blocked_by=blocked_by_task.get(str(task), ()),
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)

    high_risks = [item for item in risks if item.severity == "high"]
    top = scored[0] if scored else None
    second = scored[1] if len(scored) > 1 else None
    margin = 0.18 if top is None or second is None else max(0.0, top.score - second.score)
    confidence = min(0.95, max(0.25, (top.score if top is not None else 0.35) - 0.08 * len(high_risks) + 0.5 * margin))

    if high_risks:
        risk = high_risks[0]
        if "missingness" in risk.risk_id:
            action = RecommendedNextStep(
                action_id="review:missingness_strategy",
                title="Review missingness before taskifying",
                kind="review",
                rationale=risk.rationale,
                confidence=confidence,
                command_hint="taskify(base, task='masked_reconstruction', ...) or document the imputation policy",
                why_not=tuple(
                    f"`{item.task}` stays secondary until missingness is handled." for item in scored[:2] if item.task != "masked_reconstruction"
                ),
            )
        else:
            action = RecommendedNextStep(
                action_id="review:irregular_sampling_strategy",
                title="Review irregular sampling before taskifying",
                kind="review",
                rationale=risk.rationale,
                confidence=confidence,
                command_hint="document the observation policy or compare regular-vs-irregular task views",
                why_not=tuple(
                    f"`{item.task}` stays secondary until the observation mechanism is explicit." for item in scored[:2]
                ),
            )
    elif top is not None:
        why_not: list[str] = []
        for item in scored[1:3]:
            why_not.append(f"`{item.task}` scored lower because {item.rationale.lower()}")
        if top.blocked_by:
            why_not.insert(0, f"`{top.task}` is currently blocked by {', '.join(top.blocked_by)}")
        action = RecommendedNextStep(
            action_id=f"run:taskify:{top.task}",
            title=f"Taskify into `{top.task}`",
            kind="run",
            rationale=top.rationale,
            confidence=confidence,
            command_hint=_task_hint(top.task),
            why_not=tuple(why_not),
        )
    else:
        action = RecommendedNextStep(
            action_id="review:report_findings",
            title="Review the report and choose a downstream task",
            kind="review",
            rationale="No candidate task clearly dominates, so human review should come before taskification.",
            confidence=0.35,
            command_hint="open report.html and inspect the linked docs/examples/API suggestions",
        )

    summary = (
        "This decision record separates hard facts, risks, candidate tasks, and the single recommended next step "
        "so people and agents can see why the bundle is being routed in one direction instead of another."
    )
    return DecisionRecord(
        kind="dataset",
        dataset_id=flags["dataset_id"],
        goal=flags["goal"] or None,
        summary=summary,
        facts=facts,
        risks=tuple(risks),
        blockers=tuple(blockers),
        candidate_tasks=tuple(scored),
        recommended_next_step=action,
    )


def build_task_decision_record(context: AgentContextPack) -> DecisionRecord:
    compact = dict(context.compact)
    task = str(compact.get("task", "task"))
    facts = (
        DecisionFact("task", "Task", task),
        DecisionFact("X_shape", "X shape", str(compact.get("X_shape"))),
        DecisionFact("y_shape", "y shape", str(compact.get("y_shape"))),
        DecisionFact("mask_keys", "Mask keys", ", ".join(compact.get("mask_keys", [])) or "none"),
        DecisionFact("aux_keys", "Aux keys", ", ".join(compact.get("aux_keys", [])) or "none"),
    )
    action = RecommendedNextStep(
        action_id="review:schema",
        title="Inspect the task schema before modeling",
        kind="review",
        rationale=f"`{task}` is already taskified, so the most useful next step is to verify X/y semantics, masks, and aux keys before training.",
        confidence=0.88,
        command_hint="open task_context.json and task_card.md before loading raw arrays",
        why_not=("`train:baseline` comes second because the schema should be checked first.",),
    )
    return DecisionRecord(
        kind="task",
        dataset_id=task,
        goal=str(compact.get("goal") or "") or None,
        summary="This decision record makes the first move explicit for a task dataset: understand the schema first, then train or evaluate.",
        facts=facts,
        risks=tuple(),
        blockers=tuple(),
        candidate_tasks=(CandidateTaskDecision(task=task, score=1.0, rationale="The asset is already taskified.", blocked_by=tuple()),),
        recommended_next_step=action,
    )


__all__ = [
    "DecisionFact",
    "DecisionRisk",
    "CandidateTaskDecision",
    "RecommendedNextStep",
    "DecisionRecord",
    "build_dataset_decision_record",
    "build_task_decision_record",
]
