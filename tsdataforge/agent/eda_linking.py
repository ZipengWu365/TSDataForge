from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..analysis.dataset import DatasetDescription
from ..analysis.describe import SeriesDescription
from .api_reference import APIReference, APISymbol, build_api_reference
from .examples import ExampleRecipe, example_catalog, recommend_examples


@dataclass(frozen=True)
class FAQEntry:
    faq_id: str
    question: str
    answer: str
    keywords: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EDARoute:
    route_id: str
    title: str
    summary: str
    query_tokens: tuple[str, ...] = field(default_factory=tuple)
    recommended_tasks: tuple[str, ...] = field(default_factory=tuple)
    page_paths: tuple[str, ...] = field(default_factory=tuple)
    example_ids: tuple[str, ...] = field(default_factory=tuple)
    api_names: tuple[str, ...] = field(default_factory=tuple)
    faq_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EDAResourceLink:
    kind: str
    resource_id: str
    title: str
    path: str
    reason: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EDATaskSuggestion:
    task: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EDAResourceHub:
    kind: str
    query: str
    tags: list[str] = field(default_factory=list)
    recommended_tasks: list[EDATaskSuggestion] = field(default_factory=list)
    page_links: list[EDAResourceLink] = field(default_factory=list)
    example_links: list[EDAResourceLink] = field(default_factory=list)
    api_links: list[EDAResourceLink] = field(default_factory=list)
    faq_links: list[EDAResourceLink] = field(default_factory=list)
    routes: list[EDARoute] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "query": self.query,
            "tags": list(self.tags),
            "recommended_tasks": [item.to_dict() for item in self.recommended_tasks],
            "page_links": [item.to_dict() for item in self.page_links],
            "example_links": [item.to_dict() for item in self.example_links],
            "api_links": [item.to_dict() for item in self.api_links],
            "faq_links": [item.to_dict() for item in self.faq_links],
            "routes": [item.to_dict() for item in self.routes],
            "next_steps": list(self.next_steps),
        }

    def to_markdown(self) -> str:
        return render_eda_resource_hub_markdown(self)


DOC_PAGES: tuple[dict[str, Any], ...] = (
    {"page_id": "landing", "title": "Docs", "path": "index.html", "keywords": ("landing", "overview", "journeys", "eda", "taskify")},
    {"page_id": "quickstart", "title": "Quickstart", "path": "getting-started.html", "keywords": ("quickstart", "first steps", "eda", "series", "dataset")},
    {"page_id": "tutorials", "title": "Tutorials", "path": "tutorials.html", "keywords": ("tutorials", "learning paths", "onboarding", "eda")},
    {"page_id": "cookbook", "title": "Examples", "path": "cookbook.html", "keywords": ("examples", "cookbook", "recipes", "eda")},
    {"page_id": "taskification", "title": "Taskification", "path": "taskification.html", "keywords": ("taskify", "forecasting", "classification", "causal", "control")},
    {"page_id": "agent_playbook", "title": "Agent Playbook", "path": "agent-playbook.html", "keywords": ("agent", "context", "schema", "cards", "eda")},
    {"page_id": "use_cases", "title": "Use Cases", "path": "use-cases.html", "keywords": ("use cases", "research", "real data", "agent")},
    {"page_id": "api", "title": "API Reference", "path": "api-reference.html", "keywords": ("api", "reference", "functions", "classes", "eda")},
    {"page_id": "faq", "title": "FAQ", "path": "faq.html", "keywords": ("faq", "questions", "support", "eda")},
    {"page_id": "rollout", "title": "Adoption", "path": "rollout.html", "keywords": ("adoption", "launch", "sharing", "bundle")},
)


FAQ_ENTRIES: tuple[FAQEntry, ...] = (
    FAQEntry(
        faq_id="eda_links_how",
        question="How are the landing / docs / examples / API / FAQ links inside an EDA report generated?",
        answer="They are not hard-coded for one report. They are derived from structure tags, sampling and missingness features, trace clues, and a shared route map. The same mapping drives the report, example pages, API pages, and FAQ page.",
        keywords=("eda", "links", "routes", "examples", "api", "faq"),
    ),
    FAQEntry(
        faq_id="eda_after_report",
        question="After a real-data EDA report, how should I choose between forecasting, classification, and causal tasks?",
        answer="Start from the recommended tasks in the report, then align them with your scientific or product goal. Trend and seasonality usually suggest forecasting; multivariate coupling suggests classification or system identification; interventions, graphs, or trace clues suggest causal or counterfactual tasks.",
        keywords=("after report", "task selection", "forecasting", "classification", "causal"),
    ),
    FAQEntry(
        faq_id="missing_irregular_next",
        question="What should I do if the report flags missingness or irregular sampling?",
        answer="Model the observation mechanism explicitly first. Then decide whether masked reconstruction, robust forecasting, or regular-vs-irregular synthetic controls are the right next step. Do not expect the downstream model to absorb these issues implicitly.",
        keywords=("missing", "irregular", "masked reconstruction", "observation"),
    ),
    FAQEntry(
        faq_id="taskify_vs_generate",
        question="When should I call generate_dataset directly, and when should I use generate_series_dataset plus taskify?",
        answer="If the task is already known and you want the shortest path to X/y, call generate_dataset directly. If you want to reuse the same base data across multiple tasks or run EDA first, start with generate_series_dataset and taskify later.",
        keywords=("generate_dataset", "taskify", "seriesdataset", "taskdataset"),
    ),
    FAQEntry(
        faq_id="real_data_no_trace",
        question="Can real data still be useful without trace ground truth?",
        answer="Yes. Real data can still be described, reported, and taskified. Only tasks that require explicit ground truth, such as causal discovery with known adjacency or ITE supervision, usually need synthetic data or external labels.",
        keywords=("real data", "trace", "taskify", "causal"),
    ),
    FAQEntry(
        faq_id="compact_context_why",
        question="Why emphasize compact contexts and artifact cards?",
        answer="Because agents should not read full READMEs and raw arrays by default. Compact contexts, cards, and schema compress the key semantics into a much smaller and more stable token budget.",
        keywords=("compact context", "card", "schema", "agent"),
    ),
    FAQEntry(
        faq_id="simulator_scope",
        question="Will TSDataForge rebuild simulation stacks that already exist?",
        answer="No. External simulations or real rollouts should usually enter through wrap_external_series or SeriesDataset.from_arrays. TSDataForge focuses on taxonomy, EDA, taskification, and agent-friendly assets.",
        keywords=("simulator", "external", "wrap_external_series"),
    ),
    FAQEntry(
        faq_id="taskify_semantics",
        question="Does taskification destroy the meaning of the raw data?",
        answer="No. TaskDataset.schema makes the meaning of X, y, masks, and aux explicit, while the base SeriesDataset still keeps values, time, meta, and trace.",
        keywords=("taskify", "schema", "semantics"),
    ),
    FAQEntry(
        faq_id="public_surface",
        question="What should I publish first on the public surface?",
        answer="A landing page, a five-minute quickstart, 10–20 copyable examples, an API reference, an FAQ, and at least a few sample assets with cards and compact contexts.",
        keywords=("landing", "quickstart", "api", "faq", "examples"),
    ),
    FAQEntry(
        faq_id="agent_minimal_stack",
        question="What is the minimum useful stack for an agent pipeline?",
        answer="In most cases, build_agent_context, build_*_card, the API manifest, recommend_examples, and TaskDataset.schema are enough to build a stable first integration.",
        keywords=("agent", "pipeline", "context", "card", "schema"),
    ),
)


COMMON_EDA_ROUTES: tuple[EDARoute, ...] = (
    EDARoute(
        route_id="trend_seasonal",
        title="Trend / seasonal dominated real series",
        summary="When a report highlights trend, seasonality, dominant period, or obvious ACF structure, forecasting, decomposition, and spec matching are usually the first tasks to try.",
        query_tokens=("trend", "seasonal", "period", "dominant_period", "forecasting", "decomposition", "autocorrelation"),
        recommended_tasks=("forecasting",),
        page_paths=("getting-started.html", "taskification.html", "cookbook.html", "api-reference.html"),
        example_ids=("real_series_eda", "taskify_forecasting", "spec_from_real_series"),
        api_names=("describe_series", "generate_eda_report", "suggest_spec", "taskify_dataset", "generate_dataset"),
        faq_ids=("eda_after_report", "taskify_vs_generate"),
    ),
    EDARoute(
        route_id="missing_irregular",
        title="Missingness / irregular sampling comes first",
        summary="When the report flags missingness or irregular sampling, make the observation mechanism explicit before you choose between masked reconstruction, robust forecasting, or synthetic observation controls.",
        query_tokens=("missing", "irregular_sampling", "observation", "mask", "robust", "sampling", "dt_cv"),
        recommended_tasks=("masked_reconstruction", "forecasting"),
        page_paths=("getting-started.html", "taskification.html", "faq.html", "api-reference.html"),
        example_ids=("real_series_eda", "robust_observation_variants", "masked_reconstruction"),
        api_names=("describe_series", "generate_eda_report", "ObservationSpec", "IrregularSampling", "BlockMissing"),
        faq_ids=("missing_irregular_next", "taskify_vs_generate"),
    ),
    EDARoute(
        route_id="multivariate_coupled",
        title="Multivariate / coupled structure",
        summary="When channels are numerous and cross-channel structure is strong, classification, system identification, and causal response are often the highest-value task views.",
        query_tokens=("multivariate", "coupled", "cross correlation", "system identification", "causal", "multi-channel", "varx"),
        recommended_tasks=("classification", "system_identification", "causal_response"),
        page_paths=("taskification.html", "cookbook.html", "api-reference.html", "agent-playbook.html"),
        example_ids=("dataset_eda", "system_identification", "causal_response", "real_dataset_taskify"),
        api_names=("describe_dataset", "generate_dataset_eda_report", "LinearStateSpace", "CausalVARX", "taskify_dataset"),
        faq_ids=("eda_after_report", "real_data_no_trace"),
    ),
    EDARoute(
        route_id="bursty_events",
        title="Bursty / spikes / event-like structure",
        summary="When spikes, heavy tails, or sparse events dominate the report, anomaly detection, event detection, and intervention detection usually become more meaningful than pure forecasting.",
        query_tokens=("bursty", "spikes", "heavy_tail", "event", "anomaly", "intermittent", "pulse"),
        recommended_tasks=("anomaly_detection", "event_detection", "intervention_detection"),
        page_paths=("cookbook.html", "taskification.html", "api-reference.html", "faq.html"),
        example_ids=("anomaly_detection", "event_control_detection", "intervention_detection"),
        api_names=("generate_dataset", "BurstyPulseTrain", "EventTriggeredController", "InterventionSpec"),
        faq_ids=("eda_after_report",),
    ),
    EDARoute(
        route_id="regime_change",
        title="Regime / changepoint / switching behavior",
        summary="When means, variances, or local dynamics switch over time, change-point tasks and regime-aware benchmarks are often the right next stop.",
        query_tokens=("regime", "change", "changepoint", "switching", "piecewise", "segment"),
        recommended_tasks=("change_point_detection", "classification"),
        page_paths=("cookbook.html", "taskification.html", "api-reference.html"),
        example_ids=("change_point_detection", "classification_benchmark"),
        api_names=("RegimeSwitch", "generate_dataset", "taskify_dataset"),
        faq_ids=("eda_after_report",),
    ),
    EDARoute(
        route_id="control_policy",
        title="Control / input-output / policy-driven sequences",
        summary="If the report comes from a rollout with input, state, output, or reward, system identification, policy value, and counterfactual tasks should be considered first.",
        query_tokens=("control", "input", "state", "output", "reward", "policy", "system identification", "servo"),
        recommended_tasks=("system_identification", "policy_value_estimation", "counterfactual_response"),
        page_paths=("cookbook.html", "taskification.html", "api-reference.html", "agent-playbook.html"),
        example_ids=("system_identification", "policy_value_estimation", "policy_counterfactual"),
        api_names=("LinearStateSpace", "PolicyControlledStateSpace", "generate_counterfactual_pair", "with_policy"),
        faq_ids=("taskify_vs_generate", "agent_minimal_stack"),
    ),
    EDARoute(
        route_id="causal_intervention",
        title="Causal / intervention / counterfactual clues",
        summary="When trace or data schema contains treatment, adjacency, counterfactual outputs, or intervention masks, causal response, discovery, ITE, and counterfactual tasks usually deserve priority.",
        query_tokens=("causal", "intervention", "counterfactual", "adjacency", "treatment", "ite", "do"),
        recommended_tasks=("causal_response", "causal_discovery", "causal_ite", "counterfactual_response"),
        page_paths=("taskification.html", "cookbook.html", "api-reference.html", "faq.html"),
        example_ids=("causal_response", "policy_counterfactual", "intervention_detection"),
        api_names=("CausalVARX", "CausalTreatmentOutcome", "generate_counterfactual_pair", "with_intervention"),
        faq_ids=("real_data_no_trace", "eda_after_report"),
    ),
    EDARoute(
        route_id="dataset_inventory",
        title="Dataset inventory / coverage / structure bucketing",
        summary="When you are facing a whole collection rather than one series, run dataset-level EDA, signature statistics, and task routing before anything else.",
        query_tokens=("dataset", "inventory", "coverage", "signature", "tag_counts", "many series"),
        recommended_tasks=("classification", "forecasting"),
        page_paths=("index.html", "cookbook.html", "taskification.html", "rollout.html"),
        example_ids=("dataset_eda", "real_dataset_taskify", "dataset_cards"),
        api_names=("describe_dataset", "generate_dataset_eda_report", "SeriesDataset", "build_dataset_context"),
        faq_ids=("taskify_vs_generate", "public_surface"),
    ),
    EDARoute(
        route_id="agent_ready_assets",
        title="Turn a report into agent-ready assets",
        summary="When the goal is to hand the data and report to an agent or a team, save compact contexts, cards, API manifests, and linked docs bundles together.",
        query_tokens=("agent", "context", "cards", "manifest", "bundle", "docs", "faq"),
        recommended_tasks=(),
        page_paths=("agent-playbook.html", "api-reference.html", "rollout.html", "faq.html"),
        example_ids=("agent_context_pack", "dataset_cards", "docs_site_generation"),
        api_names=("build_agent_context", "build_series_dataset_card", "build_task_dataset_card", "generate_docs_site"),
        faq_ids=("compact_context_why", "agent_minimal_stack"),
    ),
)

def _norm_tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, dict):
        items = [str(k) for k in value.keys()] + [str(v) for v in value.values()]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value]
    else:
        items = [str(value)]
    tokens: set[str] = set()
    for item in items:
        for tok in re.findall(r"[A-Za-z0-9_./+-]+", item.lower()):
            tok = tok.strip("._/-+")
            if tok:
                tokens.add(tok)
                tokens.update(tok.replace("_", " ").split())
    return tokens


def _join_base(base: str | None, path: str) -> str:
    if not base:
        return path
    base = str(base)
    if base.endswith("/"):
        return base + path
    return base.rstrip("/") + "/" + path


def _unique_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def common_eda_finding_routes() -> tuple[EDARoute, ...]:
    return COMMON_EDA_ROUTES


def _tokens_for_route(route: EDARoute) -> set[str]:
    return _norm_tokens(route.title) | _norm_tokens(route.summary) | _norm_tokens(route.query_tokens) | _norm_tokens(route.recommended_tasks) | _norm_tokens(route.example_ids) | _norm_tokens(route.api_names)


def _route_score(query_tokens: set[str], route: EDARoute) -> float:
    route_tokens = _tokens_for_route(route)
    overlap = len(query_tokens & route_tokens)
    title_bonus = sum(1.0 for tok in query_tokens if tok in _norm_tokens(route.title))
    task_bonus = sum(0.7 for tok in query_tokens if tok in _norm_tokens(route.recommended_tasks))
    return 2.5 * overlap + title_bonus + task_bonus


def routes_for_query(query: str | Iterable[str], *, top_k: int = 4) -> list[EDARoute]:
    query_tokens = _norm_tokens(query)
    scored: list[tuple[float, int, EDARoute]] = []
    for i, route in enumerate(COMMON_EDA_ROUTES):
        scored.append((_route_score(query_tokens, route), -i, route))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    picked = [route for score, _, route in scored if score > 0][: max(1, int(top_k))]
    return picked or list(COMMON_EDA_ROUTES[: max(1, int(top_k))])


def example_eda_routes(example: ExampleRecipe, *, top_k: int = 3) -> list[EDARoute]:
    query = " ".join([
        example.title,
        example.summary,
        example.goal,
        *example.keywords,
        *example.related_api,
        example.category,
    ])
    return routes_for_query(query, top_k=top_k)


def api_eda_routes(symbol: APISymbol, *, top_k: int = 3) -> list[EDARoute]:
    query = " ".join([
        symbol.name,
        symbol.summary,
        symbol.when_to_use,
        *symbol.related,
        *symbol.example_ids,
        symbol.group,
    ])
    return routes_for_query(query, top_k=top_k)


def _series_profile(desc: SeriesDescription, trace_summary: dict[str, Any] | None) -> tuple[list[str], str, bool]:
    tags = list(desc.inferred_tags)
    if desc.n_channels > 1 and "multivariate" not in tags:
        tags.append("multivariate")
    profile_bits = [*tags]
    if desc.scores.get("dominant_period"):
        profile_bits.append("dominant_period")
    if trace_summary:
        if trace_summary.get("adjacency_shape"):
            profile_bits.extend(["causal", "adjacency"])
        if trace_summary.get("counterfactual_keys"):
            profile_bits.extend(["counterfactual", "policy"])
        if trace_summary.get("intervention_keys"):
            profile_bits.extend(["intervention", "event"])
        if trace_summary.get("episode_return") is not None:
            profile_bits.extend(["reward", "control"])
        profile_bits.extend([str(x) for x in trace_summary.get("tags", [])])
    query = " ".join(_unique_keep_order(profile_bits))
    multivariate = bool(desc.n_channels > 1)
    return tags, query, multivariate


def _dataset_profile(desc: DatasetDescription) -> tuple[list[str], str, bool]:
    tags = [k for k, _ in list(desc.tag_counts.items())[:8]]
    signatures = [k for k, _ in list(desc.signature_counts.items())[:6]]
    multivariate = float(desc.channel_stats.get("max", 1.0) or 1.0) > 1.0
    if multivariate and "multivariate" not in tags:
        tags.append("multivariate")
    query = " ".join(_unique_keep_order([*tags, *signatures, "dataset", "coverage", "inventory"]))
    return tags, query, multivariate


def _task_reason_map(tags: set[str], routes: list[EDARoute]) -> list[EDATaskSuggestion]:
    reasons: dict[str, list[str]] = {}
    for route in routes:
        for task in route.recommended_tasks:
            reasons.setdefault(task, []).append(route.title)
    ordered = []
    for task in _unique_keep_order([task for route in routes for task in route.recommended_tasks]):
        why = "; ".join(reasons.get(task, [])[:2])
        ordered.append(EDATaskSuggestion(task=task, reason=f"Matched EDA route(s): {why}."))
    if not ordered:
        if {"trend", "seasonal", "ar1_like", "random_walk_like"} & tags:
            ordered.append(EDATaskSuggestion("forecasting", "Temporal dependence is visible in the report."))
        if {"multivariate", "coupled"} & tags:
            ordered.append(EDATaskSuggestion("classification", "Multiple coupled channels usually benefit from discriminative tasks."))
    return ordered[:6]


def _page_title_from_path(path: str) -> str:
    for page in DOC_PAGES:
        if page["path"] == path:
            return str(page["title"])
    return path


def _reason_from_routes(route_titles: list[str]) -> str:
    return "Matched report findings: " + ", ".join(route_titles[:2]) + "."


def _build_page_links(routes: list[EDARoute], docs_base_url: str | None) -> list[EDAResourceLink]:
    reasons: dict[str, list[str]] = {}
    for route in routes:
        for path in route.page_paths:
            reasons.setdefault(path, []).append(route.title)
    priority = {page["path"]: i for i, page in enumerate(DOC_PAGES)}
    ordered = sorted(reasons.items(), key=lambda kv: (len(kv[1]) * -1, priority.get(kv[0], 999)))
    out: list[EDAResourceLink] = []
    for path, titles in ordered[:6]:
        out.append(
            EDAResourceLink(
                kind="page",
                resource_id=Path(path).stem,
                title=_page_title_from_path(path),
                path=_join_base(docs_base_url, path),
                reason=_reason_from_routes(titles),
                score=float(len(titles)),
            )
        )
    return out


def _api_lookup(api_ref: APIReference) -> dict[str, tuple[str, APISymbol]]:
    out: dict[str, tuple[str, APISymbol]] = {}
    for cat in api_ref.categories:
        for sym in cat.symbols:
            out[sym.name] = (cat.category_id, sym)
    return out


def _build_api_links(routes: list[EDARoute], api_ref: APIReference, docs_base_url: str | None, kind: str) -> list[EDAResourceLink]:
    base_names = ["generate_eda_report", "describe_series", "explain_series"] if kind == "series" else ["generate_dataset_eda_report", "describe_dataset"]
    if kind == "series":
        base_names += ["suggest_spec", "taskify_dataset"]
    else:
        base_names += ["SeriesDataset", "taskify_dataset"]
    route_names = [name for route in routes for name in route.api_names]
    names = _unique_keep_order([*base_names, *route_names, "build_agent_context", "generate_docs_site", "build_api_reference"])
    lookup = _api_lookup(api_ref)
    reasons: dict[str, list[str]] = {}
    for route in routes:
        for name in route.api_names:
            reasons.setdefault(name, []).append(route.title)
    out: list[EDAResourceLink] = []
    for name in names:
        if name not in lookup:
            continue
        category_id, symbol = lookup[name]
        titles = reasons.get(name, [])
        reason = _reason_from_routes(titles) if titles else symbol.when_to_use or symbol.summary
        out.append(
            EDAResourceLink(
                kind="api",
                resource_id=name,
                title=name,
                path=_join_base(docs_base_url, f"api/{category_id}.html#{name}"),
                reason=reason,
                score=float(len(titles)) if titles else 0.5,
            )
        )
    return out[:8]


def _build_example_links(routes: list[EDARoute], query: str, docs_base_url: str | None) -> list[EDAResourceLink]:
    recommended = recommend_examples(query, top_k=6)
    recommended_ids = [ex.example_id for ex in recommended]
    route_ids = [ex_id for route in routes for ex_id in route.example_ids]
    example_ids = _unique_keep_order([*route_ids, *recommended_ids])
    catalog = {ex.example_id: ex for ex in example_catalog()}
    reasons: dict[str, list[str]] = {}
    for route in routes:
        for ex_id in route.example_ids:
            reasons.setdefault(ex_id, []).append(route.title)
    out: list[EDAResourceLink] = []
    for ex_id in example_ids:
        ex = catalog.get(ex_id)
        if ex is None:
            continue
        titles = reasons.get(ex_id, [])
        reason = _reason_from_routes(titles) if titles else ex.summary
        out.append(
            EDAResourceLink(
                kind="example",
                resource_id=ex_id,
                title=ex.title,
                path=_join_base(docs_base_url, f"examples/{ex_id}.html"),
                reason=reason,
                score=float(len(titles)) if titles else 0.5,
            )
        )
    return out[:6]


def _build_faq_links(routes: list[EDARoute], query: str, docs_base_url: str | None) -> list[EDAResourceLink]:
    q_tokens = _norm_tokens(query)
    reasons: dict[str, list[str]] = {}
    for route in routes:
        for faq_id in route.faq_ids:
            reasons.setdefault(faq_id, []).append(route.title)
    scored: list[tuple[float, int, FAQEntry]] = []
    for i, faq in enumerate(FAQ_ENTRIES):
        faq_tokens = _norm_tokens(faq.question) | _norm_tokens(faq.answer) | _norm_tokens(faq.keywords)
        overlap = len(q_tokens & faq_tokens)
        route_bonus = len(reasons.get(faq.faq_id, [])) * 2.0
        scored.append((overlap + route_bonus, -i, faq))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    out: list[EDAResourceLink] = []
    for score, _, faq in scored:
        if score <= 0 and out:
            break
        titles = reasons.get(faq.faq_id, [])
        reason = _reason_from_routes(titles) if titles else faq.answer
        out.append(
            EDAResourceLink(
                kind="faq",
                resource_id=faq.faq_id,
                title=faq.question,
                path=_join_base(docs_base_url, f"faq.html#{faq.faq_id}"),
                reason=reason,
                score=float(score),
            )
        )
        if len(out) >= 5:
            break
    return out


def _next_steps(kind: str, routes: list[EDARoute], examples: list[EDAResourceLink], tasks: list[EDATaskSuggestion]) -> list[str]:
    steps: list[str] = []
    if kind == "series":
        steps.append("先把报告中的 tags、dominant period 和观测机制确认下来，再决定是否需要 synthetic matching spec。")
    else:
        steps.append("先把 top tags / signatures 看清楚，再决定是统一 taskify 还是按分桶分别处理。")
    if tasks:
        steps.append("优先尝试任务：" + ", ".join(item.task for item in tasks[:3]) + "。")
    if examples:
        steps.append("先复制最接近的案例，再把你自己的数据替换进去：" + examples[0].title + "。")
    if any(route.route_id == "agent_ready_assets" for route in routes):
        steps.append("如果要交给 agent 或团队，连同 compact context、artifact card 和 linked docs bundle 一起保存。")
    else:
        steps.append("如果要分享或沉淀知识，生成 linked docs bundle，让报告和 docs site 双向可跳转。")
    return steps


def build_eda_resource_hub(
    description: SeriesDescription | DatasetDescription,
    *,
    trace_summary: dict[str, Any] | None = None,
    docs_base_url: str | None = None,
    api_ref: APIReference | None = None,
) -> EDAResourceHub:
    """Build a report-driven navigation hub linking docs, examples, API and FAQ.

    This is the shared routing layer used by EDA reports and the static docs site.
    """

    if isinstance(description, SeriesDescription):
        kind = "series"
        tags, query, multivariate = _series_profile(description, trace_summary)
    elif isinstance(description, DatasetDescription):
        kind = "dataset"
        tags, query, multivariate = _dataset_profile(description)
    else:
        raise TypeError("description must be a SeriesDescription or DatasetDescription")

    if multivariate:
        query += " multivariate coupled"
    if trace_summary:
        query += " trace control causal"
    routes = routes_for_query(query, top_k=4)
    task_suggestions = _task_reason_map(set(tags), routes)
    api_ref = build_api_reference() if api_ref is None else api_ref
    page_links = _build_page_links(routes, docs_base_url)
    example_links = _build_example_links(routes, query, docs_base_url)
    api_links = _build_api_links(routes, api_ref, docs_base_url, kind)
    faq_links = _build_faq_links(routes, query, docs_base_url)
    steps = _next_steps(kind, routes, example_links, task_suggestions)
    return EDAResourceHub(
        kind=kind,
        query=query.strip(),
        tags=tags,
        recommended_tasks=task_suggestions,
        page_links=page_links,
        example_links=example_links,
        api_links=api_links,
        faq_links=faq_links,
        routes=routes,
        next_steps=steps,
    )


def render_eda_resource_hub_markdown(hub: EDAResourceHub) -> str:
    lines = [f"# EDA Resource Hub ({hub.kind})", "", f"Query: `{hub.query}`", ""]
    if hub.tags:
        lines.append("## Tags")
        lines.append("")
        lines.append(", ".join(f"`{tag}`" for tag in hub.tags))
        lines.append("")
    if hub.recommended_tasks:
        lines.append("## Recommended tasks")
        lines.append("")
        for item in hub.recommended_tasks:
            lines.append(f"- **{item.task}** — {item.reason}")
        lines.append("")
    for title, items in (("Docs pages", hub.page_links), ("Examples", hub.example_links), ("API", hub.api_links), ("FAQ", hub.faq_links)):
        if not items:
            continue
        lines.append(f"## {title}")
        lines.append("")
        for item in items:
            lines.append(f"- [{item.title}]({item.path}) — {item.reason}")
        lines.append("")
    if hub.next_steps:
        lines.append("## Next steps")
        lines.append("")
        for item in hub.next_steps:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def save_eda_resource_hub(hub: EDAResourceHub, output_prefix: str | Path) -> tuple[str, str]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix("")
    json_file = str(json_path) + "_resource_hub.json"
    md_file = str(json_path) + "_resource_hub.md"
    Path(json_file).write_text(json.dumps(hub.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    Path(md_file).write_text(render_eda_resource_hub_markdown(hub), encoding="utf-8")
    return json_file, md_file


__all__ = [
    "FAQEntry",
    "FAQ_ENTRIES",
    "EDARoute",
    "EDAResourceHub",
    "EDAResourceLink",
    "EDATaskSuggestion",
    "DOC_PAGES",
    "common_eda_finding_routes",
    "routes_for_query",
    "example_eda_routes",
    "api_eda_routes",
    "build_eda_resource_hub",
    "render_eda_resource_hub_markdown",
    "save_eda_resource_hub",
]
