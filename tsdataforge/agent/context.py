from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..analysis.dataset import describe_dataset
from ..analysis.describe import describe_series
from ..analysis.explain import explain_series
from ..datasets.builder import TaskDataset
from ..datasets.series_dataset import SeriesDataset
from ..series import GeneratedSeries


@dataclass
class AgentContextPack:
    kind: str
    budget: str
    compact: dict[str, Any]
    narrative: list[str] = field(default_factory=list)
    example_ids: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    estimated_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        return render_context_markdown(self)


_SERIES_SCORE_ORDER = (
    "dominant_period",
    "seasonality_power_ratio",
    "acf1_median",
    "trend_r2_median",
    "trend_slope_mean_abs",
    "cross_corr_mean_abs",
    "missing_rate",
    "dt_cv",
    "kurtosis_median",
    "spikiness_median",
)



def _safe_round(value: Any) -> Any:
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        val = float(value)
        if not np.isfinite(val):
            return None
        abs_val = abs(val)
        if abs_val >= 1000:
            return round(val, 1)
        if abs_val >= 100:
            return round(val, 2)
        if abs_val >= 1:
            return round(val, 3)
        return round(val, 4)
    if isinstance(value, np.ndarray):
        return [_safe_round(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_safe_round(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_round(v) for k, v in value.items()}
    return value



def _estimate_tokens(obj: Any) -> int:
    try:
        payload = json.dumps(obj, ensure_ascii=False)
    except Exception:
        payload = str(obj)
    return max(1, int(round(len(payload) / 4.0)))



def _trace_summary(series: GeneratedSeries | None) -> dict[str, Any]:
    if series is None or series.trace is None:
        return {}
    states = dict(series.trace.states)
    summary: dict[str, Any] = {
        "tags": list(series.trace.tags),
        "n_state_keys": int(len(states)),
    }
    for suffix, label in (
        ("/adjacency", "has_adjacency"),
        ("/reward", "has_reward"),
        ("/input", "has_input"),
        ("/state", "has_state"),
        ("/output", "has_output"),
        ("/ite", "has_ite"),
    ):
        if any(k.endswith(suffix) for k in states):
            summary[label] = True
    intervention_count = 0
    for k, v in states.items():
        if ("/intervention/" in k and k.endswith("/mask")) or k.endswith("/intervention_mask") or "/do_mask_var" in k:
            arr = np.asarray(v)
            if arr.ndim == 1:
                intervention_count += int(np.sum(arr > 0))
    if intervention_count > 0:
        summary["intervention_steps"] = intervention_count
    cf_count = len([k for k in states if "/counterfactual/" in k and k.endswith("/output")])
    if cf_count:
        summary["counterfactual_rollouts"] = cf_count
    return summary



def _budget_counts(budget: str) -> tuple[int, int]:
    budget = str(budget).lower()
    if budget in {"tiny", "xs"}:
        return 4, 3
    if budget in {"medium", "md"}:
        return 8, 6
    if budget in {"large", "lg", "full"}:
        return 12, 8
    return 6, 4



def _recommended_tasks_from_tags(tags: list[str], *, multivariate: bool = False, has_trace: bool = False) -> list[str]:
    out: list[str] = []
    tagset = set(tags)
    if {"trend", "seasonal", "random_walk_like", "ar1_like"} & tagset:
        out.append("forecasting")
    if {"bursty", "heavy_tail"} & tagset:
        out.append("anomaly_detection")
        out.append("event_detection")
    if "multivariate" in tagset or multivariate:
        out.append("classification")
        out.append("causal_response")
    if "coupled" in tagset:
        out.append("system_identification")
        out.append("causal_discovery")
    if {"missing", "irregular_sampling"} & tagset:
        out.append("masked_reconstruction")
    if has_trace:
        out.append("change_point_detection")
        out.append("counterfactual_response")
    seen: set[str] = set()
    picked: list[str] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            picked.append(item)
    return picked[:6]



def _top_scores(scores: dict[str, Any], max_items: int) -> dict[str, Any]:
    ordered_keys = [k for k in _SERIES_SCORE_ORDER if k in scores]
    for k in sorted(scores):
        if k not in ordered_keys:
            ordered_keys.append(k)
    return {k: _safe_round(scores[k]) for k in ordered_keys[:max_items]}



def build_series_context(
    values: GeneratedSeries | np.ndarray,
    time: np.ndarray | None = None,
    *,
    budget: str = "small",
    goal: str | None = None,
) -> AgentContextPack:
    """Build a compact, agent-friendly context pack for a single series."""

    if isinstance(values, GeneratedSeries):
        series = values
        arr = np.asarray(series.values, dtype=float)
        tt = np.asarray(series.time if time is None else time, dtype=float)
    else:
        series = None
        arr = np.asarray(values, dtype=float)
        if arr.ndim not in (1, 2):
            raise ValueError("Series context expects 1D/2D values or a GeneratedSeries")
        tt = np.arange(arr.shape[0], dtype=float) if time is None else np.asarray(time, dtype=float)

    desc = describe_series(arr, tt)
    expl = explain_series(desc)
    max_tags, max_bullets = _budget_counts(budget)
    trace = _trace_summary(series)
    recommended = _recommended_tasks_from_tags(desc.inferred_tags, multivariate=desc.n_channels > 1, has_trace=bool(trace))

    compact: dict[str, Any] = {
        "shape": [int(desc.length), int(desc.n_channels)],
        "sampling": {"dt_mean": _safe_round(desc.dt_mean), "dt_cv": _safe_round(desc.dt_cv)},
        "tags": list(desc.inferred_tags[:max_tags]),
        "scores": _top_scores(desc.scores, max_tags),
        "recommended_tasks": recommended,
    }
    if series is not None and series.spec is not None:
        compact["structure_id"] = series.spec.structure_id or series.spec.name
        compact["spec_tags"] = list(series.spec.tags[:max_tags])
    if trace:
        compact["trace"] = trace
    if goal:
        compact["goal"] = goal

    narrative = list(expl.bullets[:max_bullets])
    if goal:
        narrative.insert(0, f"Goal: {goal}.")

    from .examples import recommend_examples

    examples = [ex.example_id for ex in recommend_examples(" ".join([goal or "", *desc.inferred_tags, *recommended]), top_k=3)]
    pack = AgentContextPack(
        kind="series",
        budget=str(budget),
        compact=_safe_round(compact),
        narrative=narrative,
        example_ids=examples,
        next_actions=[f"try:{task}" for task in recommended[:4]] + ["render:eda_report", "save:context_json"],
    )
    pack.estimated_tokens = _estimate_tokens(pack.to_dict())
    return pack



def build_dataset_context(
    values: SeriesDataset | Any,
    time: Any | None = None,
    *,
    budget: str = "small",
    goal: str | None = None,
    max_series: int | None = 200,
) -> AgentContextPack:
    if isinstance(values, TaskDataset):
        return build_task_context(values, budget=budget, goal=goal)

    if isinstance(values, SeriesDataset):
        dataset = values
        desc = describe_dataset(dataset.values_list(), dataset.time_list(), max_series=max_series)
        dataset_id = dataset.dataset_id
        traces = [s.trace for s in dataset.series if s.trace is not None]
        has_trace = bool(traces)
    else:
        dataset = None
        desc = describe_dataset(values, time, max_series=max_series)
        dataset_id = "external_dataset"
        has_trace = False

    max_tags, max_bullets = _budget_counts(budget)
    top_tags = list(desc.tag_counts.items())[:max_tags]
    signatures = list(desc.signature_counts.items())[:max_tags]
    recommended = _recommended_tasks_from_tags([k for k, _ in top_tags], multivariate=desc.channel_stats.get("max", 1) > 1, has_trace=has_trace)

    compact: dict[str, Any] = {
        "dataset_id": dataset_id,
        "n_series": int(desc.n_series),
        "length_median": _safe_round(desc.length_stats.get("median")),
        "channels_median": _safe_round(desc.channel_stats.get("median")),
        "missing_rate_mean": _safe_round(desc.missing_rate_stats.get("mean")),
        "dt_cv_mean": _safe_round(desc.dt_cv_stats.get("mean")),
        "top_tags": [{"tag": tag, "count": int(count)} for tag, count in top_tags],
        "top_signatures": [{"signature": sig, "count": int(count)} for sig, count in signatures],
        "recommended_tasks": recommended,
        "has_trace": has_trace,
    }
    if goal:
        compact["goal"] = goal

    narrative = [
        f"Dataset has {int(desc.n_series)} series; median length≈{_safe_round(desc.length_stats.get('median'))}, median channels≈{_safe_round(desc.channel_stats.get('median'))}.",
        f"Most common tags: {', '.join([tag for tag, _ in top_tags]) or 'none'}.",
        f"Most common signatures: {', '.join([sig for sig, _ in signatures]) or 'none'}.",
        "Use the base dataset for report/handoff first, then taskify into forecasting, classification, causal, or control tasks.",
    ]
    if goal:
        narrative.insert(0, f"Goal: {goal}.")
    narrative = narrative[:max_bullets]

    from .examples import recommend_examples

    examples = [ex.example_id for ex in recommend_examples(" ".join([goal or "", *(tag for tag, _ in top_tags)]), top_k=4)]
    pack = AgentContextPack(
        kind="dataset",
        budget=str(budget),
        compact=_safe_round(compact),
        narrative=narrative,
        example_ids=examples,
        next_actions=["run:handoff", "run:report", "run:describe_dataset"] + [f"taskify:{task}" for task in recommended[:4]],
    )
    pack.estimated_tokens = _estimate_tokens(pack.to_dict())
    return pack



def build_task_context(
    dataset: TaskDataset,
    *,
    budget: str = "small",
    goal: str | None = None,
) -> AgentContextPack:
    max_tags, max_bullets = _budget_counts(budget)
    x_shape = list(np.asarray(dataset.X, dtype=object).shape)
    y_shape = None if dataset.y is None else list(np.asarray(dataset.y, dtype=object).shape)
    compact: dict[str, Any] = {
        "task": dataset.task,
        "n_samples": int(len(dataset.X)),
        "X_shape": x_shape,
        "y_shape": y_shape,
        "label_names": None if dataset.label_names is None else list(dataset.label_names[:max_tags]),
        "mask_keys": sorted((dataset.masks or {}).keys())[:max_tags],
        "aux_keys": sorted((dataset.aux or {}).keys())[:max_tags],
        "schema": _safe_round(dataset.schema or {}),
    }
    if goal:
        compact["goal"] = goal

    task_notes = {
        "forecasting": "X is history, y is future horizon; optimize for horizon-aware error metrics.",
        "classification": "X is a full sequence and y is a structure/class label.",
        "anomaly_detection": "y is an anomaly mask aligned to X time steps.",
        "change_point_detection": "y marks changepoint positions over time.",
        "event_detection": "y marks event/trigger activity over time.",
        "system_identification": "X concatenates past [u, y]; y is future output window; aux may include u/x full traces.",
        "causal_response": "X is historical multivariate context, y is future outcome channel.",
        "counterfactual_response": "y stores future counterfactual outcome(s); compare against factual rollout in X/aux.",
        "policy_value_estimation": "y is a scalar discounted return; X is rollout context.",
        "causal_discovery": "y is an adjacency matrix.",
        "causal_ite": "y is future individual treatment effect sequence.",
    }
    narrative = [
        task_notes.get(dataset.task, "TaskDataset packs X/y/schema so training code and agents can consume a stable protocol."),
        f"Primary arrays: X{tuple(x_shape)} and y{tuple(y_shape) if y_shape is not None else 'None'}.",
        f"Available masks: {', '.join(compact['mask_keys']) or 'none'}; aux: {', '.join(compact['aux_keys']) or 'none'}.",
        "Prefer reading the compact schema/card before loading raw arrays into model context.",
    ]
    if goal:
        narrative.insert(0, f"Goal: {goal}.")
    narrative = narrative[:max_bullets]

    from .examples import recommend_examples

    examples = [ex.example_id for ex in recommend_examples(" ".join([goal or "", dataset.task, *(compact['aux_keys'] or []), *(compact['mask_keys'] or [])]), top_k=3)]
    next_actions = ["run:handoff", "inspect:schema", "save:task_card", "train:baseline"]
    if dataset.task in {"forecasting", "causal_response", "counterfactual_response"}:
        next_actions.append("evaluate:horizon_metrics")
    if dataset.task in {"classification", "contrastive"}:
        next_actions.append("inspect:label_names")
    pack = AgentContextPack(
        kind="task",
        budget=str(budget),
        compact=compact,
        narrative=narrative,
        example_ids=examples,
        next_actions=next_actions,
    )
    pack.estimated_tokens = _estimate_tokens(pack.to_dict())
    return pack



def build_agent_context(
    obj: Any,
    time: Any | None = None,
    *,
    budget: str = "small",
    goal: str | None = None,
) -> AgentContextPack:
    if isinstance(obj, TaskDataset):
        return build_task_context(obj, budget=budget, goal=goal)
    if isinstance(obj, SeriesDataset):
        return build_dataset_context(obj, budget=budget, goal=goal)
    if isinstance(obj, GeneratedSeries):
        return build_series_context(obj, budget=budget, goal=goal)

    arr = np.asarray(obj)
    if arr.ndim == 1:
        return build_series_context(arr, time, budget=budget, goal=goal)
    if arr.ndim == 2:
        if time is not None and np.asarray(time).ndim == 1 and len(np.asarray(time)) == arr.shape[0]:
            return build_series_context(arr, time, budget=budget, goal=goal)
        return build_dataset_context(arr, time, budget=budget, goal=goal)
    if arr.ndim == 3:
        return build_dataset_context(arr, time, budget=budget, goal=goal)
    raise TypeError(f"Unsupported object for agent context: {type(obj)!r}")



def render_context_markdown(pack: AgentContextPack) -> str:
    compact_json = json.dumps(_safe_round(pack.compact), ensure_ascii=False, indent=2)
    bullets = "\n".join([f"- {item}" for item in pack.narrative])
    examples = ", ".join(pack.example_ids) if pack.example_ids else "none"
    actions = ", ".join(pack.next_actions) if pack.next_actions else "none"
    return (
        f"# TSDataForge {pack.kind} context ({pack.budget})\n\n"
        f"Estimated tokens: ~{pack.estimated_tokens}\n\n"
        f"## Compact summary\n```json\n{compact_json}\n```\n\n"
        f"## What matters\n{bullets or '- no narrative'}\n\n"
        f"## Suggested examples\n{examples}\n\n"
        f"## Suggested next actions\n{actions}\n"
    )
