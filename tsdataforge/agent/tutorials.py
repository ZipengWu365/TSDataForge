from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

from .examples import ExampleRecipe, example_catalog


@dataclass(frozen=True)
class TutorialTrack:
    tutorial_id: str
    title: str
    summary: str
    audience: tuple[str, ...] = field(default_factory=tuple)
    estimated_minutes: int = 20
    steps: tuple[str, ...] = field(default_factory=tuple)
    example_ids: tuple[str, ...] = field(default_factory=tuple)
    outcomes: tuple[str, ...] = field(default_factory=tuple)
    title_zh: str = ""
    summary_zh: str = ""
    steps_zh: tuple[str, ...] = field(default_factory=tuple)
    outcomes_zh: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self, *, language: str = "en") -> dict:
        lang = (language or "en").lower()
        data = asdict(self)
        if lang.startswith("zh"):
            data["title"] = self.title_zh or self.title
            data["summary"] = self.summary_zh or self.summary
            data["steps"] = list(self.steps_zh or self.steps)
            data["outcomes"] = list(self.outcomes_zh or self.outcomes)
        else:
            data["steps"] = list(self.steps)
            data["outcomes"] = list(self.outcomes)
        return data


_TUTORIALS: tuple[TutorialTrack, ...] = (
    TutorialTrack(
        tutorial_id="first-five-minutes",
        title="Your first five minutes with TSDataForge",
        summary="Generate one sequence, inspect trace, then save a task dataset without touching a training loop.",
        audience=("new_user", "researcher", "agent"),
        estimated_minutes=5,
        steps=(
            "Generate a univariate series from trend + seasonality + noise.",
            "Inspect tags and trace to see what the compiler preserved.",
            "Generate a small base dataset and taskify it into forecasting.",
            "Save the task dataset so context, cards, and README travel with it.",
        ),
        example_ids=("quickstart_univariate", "quickstart_dataset_pipeline", "taskify_forecasting"),
        outcomes=("GeneratedSeries", "TaskDataset", "saved artifacts"),
        title_zh="TSDataForge 的前 5 分钟",
        summary_zh="生成一条序列、查看 trace，再保存一个任务数据集，不需要先写训练循环。",
        steps_zh=(
            "先用 trend + seasonal + noise 生成一条单变量序列。",
            "查看 tags 和 trace，理解编译器保留了什么信息。",
            "生成一个基础数据集，再 taskify 成 forecasting。",
            "把任务数据集保存下来，让 context、card 和 README 一起落盘。",
        ),
        outcomes_zh=("GeneratedSeries", "TaskDataset", "保存后的资产"),
    ),
    TutorialTrack(
        tutorial_id="real-data-to-first-model",
        title="From a real time series to the first modeling decision",
        summary="Use describe + EDA to decide whether a dataset looks forecasting-first, classification-first, or causal/control-oriented.",
        audience=("analyst", "researcher", "maintainer"),
        estimated_minutes=15,
        steps=(
            "Wrap or load real data into TSDataForge containers.",
            "Generate a series-level or dataset-level EDA report.",
            "Read tags, coverage, and route suggestions from the report.",
            "Choose the first task and jump to the nearest runnable example.",
        ),
        example_ids=("real_series_eda", "dataset_eda", "spec_from_real_series", "real_dataset_taskify"),
        outcomes=("EDA report", "task suggestion", "spec seed"),
        title_zh="从真实时序到第一个建模决策",
        summary_zh="用 describe + EDA 决定数据更适合 forecasting、classification，还是 causal/control 方向。",
        steps_zh=(
            "先把真实数据包装进 TSDataForge 容器。",
            "生成单条或数据集级 EDA 报告。",
            "阅读 tags、覆盖率和 route 建议。",
            "确定第一个任务，再跳到最近的 runnable example。",
        ),
        outcomes_zh=("EDA 报告", "任务建议", "spec seed"),
    ),
    TutorialTrack(
        tutorial_id="one-base-dataset-many-tasks",
        title="One base dataset, many tasks",
        summary="Build one reusable SeriesDataset, then derive forecasting, classification, anomaly, and SSL tasks from the same underlying data.",
        audience=("researcher", "benchmark", "agent"),
        estimated_minutes=20,
        steps=(
            "Generate a base dataset from multiple structures.",
            "Taskify it into forecasting and classification first.",
            "Derive at least one dense-label task such as anomaly or changepoint detection.",
            "Export cards and schema so the task datasets are self-explaining.",
        ),
        example_ids=("taskify_forecasting", "classification_benchmark", "anomaly_detection", "change_point_detection", "masked_reconstruction"),
        outcomes=("reusable SeriesDataset", "multiple TaskDataset variants"),
        title_zh="一份基础数据集，派生多个任务",
        summary_zh="先建一个可复用的 SeriesDataset，再派生 forecasting、classification、anomaly、SSL 等任务。",
        steps_zh=(
            "先从多个结构生成一份基础数据集。",
            "先 taskify 成 forecasting 和 classification。",
            "再派生至少一个 dense-label 任务，比如 anomaly 或 changepoint detection。",
            "导出 card 和 schema，让任务数据集自解释。",
        ),
        outcomes_zh=("可复用的 SeriesDataset", "多个 TaskDataset 变体"),
    ),
    TutorialTrack(
        tutorial_id="control-causal-counterfactual",
        title="Control, causality, and counterfactual workflows",
        summary="Move from structured rollouts to system identification, causal response, counterfactual comparison, and intervention detection.",
        audience=("control", "causal", "ml_engineer"),
        estimated_minutes=25,
        steps=(
            "Start with an input-output or policy-controlled generator.",
            "Generate a system identification task from the same rollout family.",
            "Build a factual vs counterfactual comparison pair.",
            "Decide whether the next task should be causal response, intervention detection, or policy value estimation.",
        ),
        example_ids=("system_identification", "causal_response", "policy_counterfactual", "intervention_detection", "policy_value_estimation"),
        outcomes=("system ID task", "counterfactual pair", "policy-aware task datasets"),
        title_zh="控制、因果与反事实工作流",
        summary_zh="从结构化 rollout 走到 system identification、causal response、counterfactual 比较和 intervention detection。",
        steps_zh=(
            "先从 input-output 或 policy-controlled 生成器开始。",
            "从同一类 rollout 生成 system identification 任务。",
            "构造 factual vs counterfactual 的成对比较。",
            "决定下一步更适合 causal response、intervention detection 还是 policy value estimation。",
        ),
        outcomes_zh=("system ID 任务", "counterfactual pair", "带 policy 的任务数据集"),
    ),
    TutorialTrack(
        tutorial_id="agent-friendly-assets",
        title="Build agent-friendly assets, not giant prompts",
        summary="Turn datasets into compact contexts, dataset cards, API manifests, and example routes that agents can actually use.",
        audience=("agent", "platform", "maintainer"),
        estimated_minutes=15,
        steps=(
            "Build a compact context for a series, dataset, or task.",
            "Save dataset cards together with the data asset.",
            "Export an API manifest and point the agent to the smallest relevant example set.",
            "Use schema and cards as the default grounding layer in your agent workflow.",
        ),
        example_ids=("agent_context_pack", "dataset_cards", "api_reference_overview", "example_routing"),
        outcomes=("compact context", "ArtifactCard", "API manifest"),
        title_zh="构建 agent 友好型资产，而不是超长 prompt",
        summary_zh="把数据集变成 compact context、dataset card、API manifest 和 example route，让 agent 真正能用。",
        steps_zh=(
            "为 series、dataset 或 task 构建 compact context。",
            "把 dataset card 和数据资产一起保存。",
            "导出 API manifest，并把 agent 指向最小相关案例集。",
            "在 agent 工作流里把 schema 和 card 作为默认 grounding 层。",
        ),
        outcomes_zh=("compact context", "ArtifactCard", "API manifest"),
    ),
    TutorialTrack(
        tutorial_id="launch-and-community",
        title="Ship docs, examples, and support assets that can scale",
        summary="Generate a bilingual docs site, publish linked EDA bundles, and prepare assets that can absorb global traffic.",
        audience=("maintainer", "community", "pm"),
        estimated_minutes=20,
        steps=(
            "Generate the docs site and verify quickstart, tutorials, API, and FAQ coverage.",
            "Bundle a linked EDA report together with the docs site for sharing.",
            "Check that saved datasets include README, cards, and contexts.",
            "Publish a clear launch checklist, examples by persona, and issue routing guidance.",
        ),
        example_ids=("docs_site_generation", "dataset_cards", "api_reference_overview", "example_routing"),
        outcomes=("docs site", "launch checklist", "shareable bundles"),
        title_zh="发布能接住流量的文档、案例和支持资产",
        summary_zh="生成双语 docs site、发布 linked EDA bundle，并准备能承接全球流量的资产。",
        steps_zh=(
            "生成 docs site，并检查 quickstart、tutorials、API、FAQ 是否覆盖完整。",
            "把 linked EDA report 和 docs site 一起打包分享。",
            "检查保存的数据集是否自带 README、card 和 context。",
            "发布清晰的 launch checklist、按角色分层的案例，以及 issue 路由说明。",
        ),
        outcomes_zh=("docs site", "launch checklist", "可分享 bundle"),
    ),
)


def tutorial_catalog(*, language: str = "en") -> list[TutorialTrack]:
    lang = (language or "en").lower()
    if lang.startswith("zh"):
        localized: list[TutorialTrack] = []
        for item in _TUTORIALS:
            localized.append(
                TutorialTrack(
                    tutorial_id=item.tutorial_id,
                    title=item.title_zh or item.title,
                    summary=item.summary_zh or item.summary,
                    audience=item.audience,
                    estimated_minutes=item.estimated_minutes,
                    steps=tuple(item.steps_zh or item.steps),
                    example_ids=item.example_ids,
                    outcomes=tuple(item.outcomes_zh or item.outcomes),
                    title_zh=item.title_zh,
                    summary_zh=item.summary_zh,
                    steps_zh=item.steps_zh,
                    outcomes_zh=item.outcomes_zh,
                )
            )
        return localized
    return list(_TUTORIALS)



def _norm_tokens(text: str | Iterable[str]) -> set[str]:
    if isinstance(text, str):
        raw = text.lower().replace('-', ' ').replace('_', ' ')
        for ch in ',.;:/\\()[]{}\n\t':
            raw = raw.replace(ch, ' ')
        return {tok for tok in raw.split() if tok}
    out: set[str] = set()
    for item in text:
        out |= _norm_tokens(str(item))
    return out



def recommend_tutorials(query: str, *, top_k: int = 4, language: str = "en") -> list[TutorialTrack]:
    q = _norm_tokens(query)
    catalog = tutorial_catalog(language=language)
    examples = {ex.example_id: ex for ex in example_catalog(language=language)}
    scored: list[tuple[float, int, TutorialTrack]] = []
    for i, item in enumerate(catalog):
        tokens = _norm_tokens(item.title) | _norm_tokens(item.summary) | _norm_tokens(item.audience) | _norm_tokens(item.steps) | _norm_tokens(item.outcomes)
        for ex_id in item.example_ids:
            ex = examples.get(ex_id)
            if ex is not None:
                tokens |= _norm_tokens((ex.title, ex.summary, ex.goal, ex.keywords))
        overlap = len(q & tokens)
        title_bonus = sum(1.5 for tok in q if tok in _norm_tokens(item.title))
        score = 3.0 * overlap + title_bonus
        scored.append((score, -i, item))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    picked = [item for score, _, item in scored if score > 0][: max(1, int(top_k))]
    if picked:
        return picked
    return list(catalog[: max(1, int(top_k))])



_TUTORIALS = _TUTORIALS + (
    TutorialTrack(
        tutorial_id="live-signals-and-similarity",
        title="Live signals and similarity: turn public attention and market data into case studies",
        summary="Fetch public GitHub, crypto, and commodity series; generate an EDA report; and publish a similarity-backed story people can actually share.",
        audience=("researcher", "community", "maintainer", "agent"),
        estimated_minutes=20,
        steps=(
            "Fetch a live series such as GitHub stars, Bitcoin, gold, or oil.",
            "Generate an EDA report so the series can explain itself before you compare it.",
            "Run pairwise similarity or ranked matching on the aligned series panel.",
            "Save the report, similarity outputs, and docs bundle as a shareable case study.",
        ),
        example_ids=("openclaw_stars_similarity", "github_stars_pairwise_panel", "btc_gold_oil_similarity"),
        outcomes=("live-series artifact", "similarity report", "shareable case-study bundle"),
        title_zh="实时信号与相似性：把公共关注度和市场数据做成案例",
        summary_zh="抓取 GitHub、加密货币和大宗商品的公共时间序列，生成 EDA 报告，并发布带相似性证据的可分享案例。",
        steps_zh=(
            "抓取 GitHub stars、Bitcoin、黄金或原油等实时序列。",
            "先生成 EDA 报告，让序列先解释自己，再进入比较阶段。",
            "对齐后做 pairwise similarity 或 ranked matching。",
            "把报告、相似性输出和 docs bundle 一起保存成可分享案例。",
        ),
        outcomes_zh=("实时序列资产", "相似性报告", "可分享案例 bundle"),
    ),
)
