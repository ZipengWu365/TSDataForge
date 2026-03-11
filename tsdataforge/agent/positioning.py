from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PackageProfile:
    package_id: str
    title: str
    official_url: str
    kind: str = "companion"
    one_liner: str = ""
    primary_strengths: tuple[str, ...] = field(default_factory=tuple)
    best_for: tuple[str, ...] = field(default_factory=tuple)
    not_the_main_job: tuple[str, ...] = field(default_factory=tuple)
    tsdataforge_difference: str = ""
    combine_pattern: str = ""
    agent_token_story: str = ""
    environment_ids: tuple[str, ...] = field(default_factory=tuple)
    keywords: tuple[str, ...] = field(default_factory=tuple)
    one_liner_zh: str = ""
    tsdataforge_difference_zh: str = ""
    combine_pattern_zh: str = ""
    agent_token_story_zh: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PositioningMatrix:
    version: str
    thesis: str
    thesis_zh: str
    profiles: tuple[PackageProfile, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "thesis": self.thesis,
            "thesis_zh": self.thesis_zh,
            "profiles": [p.to_dict() for p in self.profiles],
        }

    def to_markdown(self, *, language: str = "en") -> str:
        zh = language.startswith("zh")
        lines = [
            f"# TSDataForge ecosystem positioning v{self.version}" if not zh else f"# TSDataForge 生态定位 v{self.version}",
            "",
            self.thesis if not zh else self.thesis_zh,
            "",
            "| Library | Best at | How TSDataForge differs | Use together when |" if not zh else "| 库 | 最擅长什么 | TSDataForge 的差异 | 适合一起用的时候 |",
            "|---|---|---|---|",
        ]
        for item in self.profiles:
            best = "; ".join(item.best_for[:3]) if item.best_for else "—"
            diff = item.tsdataforge_difference if not zh else (item.tsdataforge_difference_zh or item.tsdataforge_difference)
            combine = item.combine_pattern if not zh else (item.combine_pattern_zh or item.combine_pattern)
            title = f"[{item.title}]({item.official_url})"
            lines.append(f"| {title} | {best} | {diff} | {combine} |")
        lines += [
            "",
            "## Reading guide" if not zh else "## 阅读指南",
            "",
            "TSDataForge is not trying to replace the forecasting/modeling/toolbox ecosystem. Its job is to sit between raw time-series datasets and downstream libraries so the data is easier to explain, taskify, save, compare, and hand off."
            if not zh else
            "TSDataForge 并不是要替代 forecasting / modeling / toolbox 生态。它的职责是贴着原始或仿真的时序资产工作，让这些资产更容易被解释、任务化、保存、比较，并交给人或 agent。",
            "",
        ]
        return "\n".join(lines).strip() + "\n"


_BASE_PROFILES = [
    PackageProfile(
        package_id="tsdataforge",
        title="TSDataForge",
        official_url="https://github.com/ZipengWu365/TSDataForge",
        kind="self",
        one_liner="A time-series profiling and handoff layer for reports, cards, contexts, decision records, and reusable bundles before modeling.",
        primary_strengths=(
            "time-series profiling before model choice",
            "report + card + context + decision record packaging",
            "one base dataset -> many task datasets",
            "compact handoff artifacts for people and automation",
            "public docs and shareable bundles",
        ),
        best_for=(
            "understanding raw time-series datasets before modeling",
            "handing datasets to teammates or automation without raw arrays",
            "deriving forecasting/classification/causal/control views from one base dataset",
            "shipping compact, self-explaining artifacts",
        ),
        not_the_main_job=(
            "large-scale simulator replacement",
            "model zoo for forecasting",
            "single-purpose feature extraction",
        ),
        tsdataforge_difference="This is the reference row: TSDataForge focuses on the profiling and handoff layer between raw datasets and downstream model libraries.",
        combine_pattern="Use it as the data-understanding and handoff layer, then pass task datasets to forecasting, classification, or similarity libraries.",
        agent_token_story="Contexts, cards, schemas, and docs bundles make automation workflows smaller, more stable, and easier to audit.",
        environment_ids=("jupyter-notebook", "python-script", "llm-agent-workflow", "static-docs-site"),
        keywords=("asset layer", "EDA", "taskification", "agent", "docs"),
        one_liner_zh="一个 structure-first 的时间序列资产层：spec、EDA、taskification、cards、docs 和 agent-ready contexts。",
        tsdataforge_difference_zh="这是参考行：TSDataForge 关注的是序列资产与下游模型库之间的那一层。",
        combine_pattern_zh="先把它当成资产与解释层，再把 task dataset 交给 forecasting、classification 或 similarity 库。",
        agent_token_story_zh="contexts、cards、schema 和 docs bundle 能让 agent 工作流更省 token、更稳定、更容易审计。",
    ),
    PackageProfile(
        package_id="tsfresh",
        title="tsfresh",
        official_url="https://tsfresh.readthedocs.io/",
        one_liner="Automated extraction and relevance filtering of large numbers of time-series characteristics.",
        primary_strengths=("feature extraction", "feature relevance", "feature-based modeling"),
        best_for=("turning sequences into tabular features", "feature-based classification or regression", "interpretable engineered descriptors"),
        not_the_main_job=("linked EDA reports", "base-dataset -> many-task asset workflows", "agent-oriented cards and contexts"),
        tsdataforge_difference="TSDataForge is not feature-extraction-first; it is asset-first. It keeps sequence semantics, task schemas, and report surfaces intact before or alongside feature extraction.",
        combine_pattern="Use TSDataForge to profile, taskify, and save the dataset asset, then run tsfresh on the task view or the base dataset when handcrafted descriptors are the right modeling interface.",
        agent_token_story="tsfresh helps summarize sequences numerically, while TSDataForge adds low-token context about why the data exists, what task it serves, and how it should be consumed.",
        environment_ids=("python-script", "jupyter-notebook"),
        keywords=("feature extraction", "characteristics", "classification", "regression", "tabular"),
        one_liner_zh="自动提取并筛选大量时间序列特征。",
        tsdataforge_difference_zh="TSDataForge 不是 feature-extraction-first，而是 asset-first。它会在特征提取之前或同时保留序列语义、任务 schema 和报告表层。",
        combine_pattern_zh="先用 TSDataForge 做 profile、taskify 和资产保存；当建模接口更适合手工特征时，再把 task view 或 base dataset 交给 tsfresh。",
        agent_token_story_zh="tsfresh 负责数值化概括序列，而 TSDataForge 会补上低 token 的上下文：数据为什么存在、服务什么任务、应该怎样被消费。",
    ),
    PackageProfile(
        package_id="sktime",
        title="sktime",
        official_url="https://www.sktime.net/",
        one_liner="A unified framework for machine learning with time series.",
        primary_strengths=("unified estimator interface", "time-series ML tasks", "ecosystem interoperability"),
        best_for=("training/evaluating estimators", "time-series pipelines", "scikit-learn-like workflows"),
        not_the_main_job=("HTML EDA reports linked to docs/examples/FAQ", "compact agent contexts", "public artifact cards"),
        tsdataforge_difference="TSDataForge is earlier and broader in the workflow: define/import structure, explain the data, derive tasks, then hand off to estimator frameworks such as sktime.",
        combine_pattern="Use TSDataForge to prepare and explain the dataset asset, and sktime to fit estimators on the resulting task dataset.",
        agent_token_story="sktime organizes estimators well; TSDataForge adds smaller and more explicit data-facing artifacts for agent handoff and documentation.",
        environment_ids=("python-script", "jupyter-notebook", "ci-pipeline"),
        keywords=("estimators", "forecasting", "classification", "pipelines", "interoperability"),
        one_liner_zh="一个做时间序列机器学习的统一框架。",
        tsdataforge_difference_zh="TSDataForge 位于更前面也更宽：先定义/导入结构、解释数据、派生任务，再交给 sktime 这样的 estimator 框架。",
        combine_pattern_zh="先用 TSDataForge 准备并解释数据资产，再把得到的 task dataset 交给 sktime 训练 estimator。",
        agent_token_story_zh="sktime 很擅长组织 estimator；TSDataForge 则补上更小、更显式的数据侧资产，方便 agent 交接和文档化。",
    ),
    PackageProfile(
        package_id="darts",
        title="Darts",
        official_url="https://unit8co.github.io/darts/",
        one_liner="A user-friendly library for forecasting and anomaly detection on time series.",
        primary_strengths=("forecasting models", "anomaly detection", "backtesting", "covariates"),
        best_for=("forecast model experimentation", "anomaly workflows", "user-friendly model APIs"),
        not_the_main_job=("spec-first synthetic benchmarks", "taskification across many task families", "agent-ready documentation bundles"),
        tsdataforge_difference="TSDataForge is not a forecast-model zoo. It focuses on structure-aware assets, explainable EDA, and deriving stable task datasets that can later be modeled in Darts.",
        combine_pattern="Use TSDataForge to build or profile the dataset, then pass the forecasting or anomaly task view into Darts models and backtesting utilities.",
        agent_token_story="Darts shines at modeling workflows; TSDataForge contributes the lighter-weight context, cards, and report layer that agents and maintainers need around those workflows.",
        environment_ids=("jupyter-notebook", "python-script"),
        keywords=("forecasting", "anomaly", "backtesting", "covariates", "models"),
        one_liner_zh="一个面向 forecasting 和 anomaly detection 的易用库。",
        tsdataforge_difference_zh="TSDataForge 不是 forecast-model zoo。它更关注结构感知资产、可解释 EDA，以及先派生稳定 task dataset，再把它交给 Darts。",
        combine_pattern_zh="先用 TSDataForge 构建或 profile 数据集，再把 forecasting 或 anomaly task view 交给 Darts 模型和回测工具。",
        agent_token_story_zh="Darts 很强在建模流程；TSDataForge 补的是 agent 和维护者需要的轻量上下文、cards 和报告层。",
    ),
    PackageProfile(
        package_id="tslearn",
        title="tslearn",
        official_url="https://tslearn.readthedocs.io/",
        one_liner="A machine learning toolkit dedicated to time-series data.",
        primary_strengths=("time-series clustering", "time-series classification", "DTW and related metrics"),
        best_for=("shape-based similarity", "clustering", "distance-based methods"),
        not_the_main_job=("dataset cards", "linked EDA reports", "public docs bundles for assets"),
        tsdataforge_difference="TSDataForge uses similarity as one tool, not the whole package thesis. It wraps similarity inside broader structure, task, and explanation workflows.",
        combine_pattern="Use TSDataForge to define or inspect the dataset and task semantics, then use tslearn when DTW-style metrics, clustering, or shape-based learners are the main downstream step.",
        agent_token_story="tslearn provides algorithms; TSDataForge provides the surrounding explanation layer so an agent can know what the arrays mean before calling those algorithms.",
        environment_ids=("jupyter-notebook", "python-script"),
        keywords=("DTW", "clustering", "classification", "shape", "distance"),
        one_liner_zh="一个专门面向时间序列机器学习的工具包。",
        tsdataforge_difference_zh="TSDataForge 把 similarity 当成其中一个工具，而不是整个包的中心命题。它把 similarity 放在更大的结构、任务和解释工作流里。",
        combine_pattern_zh="先用 TSDataForge 明确数据集和任务语义；当 DTW 类度量、聚类或 shape-based learner 才是主步骤时，再交给 tslearn。",
        agent_token_story_zh="tslearn 提供算法；TSDataForge 提供外围解释层，让 agent 在调用这些算法前先知道数组语义。",
    ),
    PackageProfile(
        package_id="gluonts",
        title="GluonTS",
        official_url="https://ts.gluon.ai/",
        one_liner="A package for probabilistic time-series modeling, focusing on deep learning based models.",
        primary_strengths=("probabilistic forecasting", "deep-learning models", "distributional outputs"),
        best_for=("probabilistic forecasts", "deep-learning forecasting research", "distribution-aware evaluation"),
        not_the_main_job=("EDA and reporting", "taskification across non-forecast tasks", "agent-friendly compact metadata"),
        tsdataforge_difference="TSDataForge is model-agnostic and asset-centric; GluonTS is model-centric and forecasting-centric. They operate at different layers of the workflow.",
        combine_pattern="Use TSDataForge to create an explained forecasting asset and task schema, then feed that forecasting task into GluonTS when probabilistic modeling is the goal.",
        agent_token_story="GluonTS gives you probabilistic model infrastructure; TSDataForge gives you smaller, clearer handoff objects that explain the data contract around it.",
        environment_ids=("python-script", "headless-server", "jupyter-notebook"),
        keywords=("probabilistic", "forecasting", "deep learning", "distribution"),
        one_liner_zh="一个以深度学习模型为主的概率时间序列建模包。",
        tsdataforge_difference_zh="TSDataForge 是 model-agnostic 且 asset-centric；GluonTS 是 model-centric 且 forecasting-centric。两者位于工作流的不同层。",
        combine_pattern_zh="先用 TSDataForge 生成带解释的 forecasting 资产和 task schema；当目标是概率建模时，再把 forecasting task 交给 GluonTS。",
        agent_token_story_zh="GluonTS 给你概率模型基础设施；TSDataForge 给你更小、更清楚的交接对象，用来解释围绕它的数据契约。",
    ),
    PackageProfile(
        package_id="aeon",
        title="aeon",
        official_url="https://www.aeon-toolkit.org/",
        one_liner="A scikit-learn compatible toolkit for time-series machine-learning tasks such as classification, regression, clustering, anomaly detection, and segmentation.",
        primary_strengths=("task breadth", "time-series ML algorithms", "benchmarking datasets"),
        best_for=("classification/regression/clustering", "benchmark evaluation", "algorithm comparison"),
        not_the_main_job=("EDA report routing", "agent asset packaging", "public docs bundles centered on one dataset asset"),
        tsdataforge_difference="TSDataForge leans harder into the data-asset and explanation layer: spec, real-data profiling, taskification, cards, and bundle export. aeon leans into algorithm/toolbox breadth.",
        combine_pattern="Use TSDataForge to create, route, and save the dataset asset, then use aeon algorithms or benchmark formats when the evaluation layer matters most.",
        agent_token_story="aeon helps with algorithm-side breadth; TSDataForge helps agents understand and move the data-side artifact through the pipeline.",
        environment_ids=("python-script", "jupyter-notebook", "ci-pipeline"),
        keywords=("classification", "clustering", "anomaly", "segmentation", "benchmarking"),
        one_liner_zh="一个与 scikit-learn 兼容的时间序列机器学习工具包，覆盖 classification、regression、clustering、anomaly detection 和 segmentation。",
        tsdataforge_difference_zh="TSDataForge 更偏数据资产与解释层：spec、真实数据 profiling、taskification、cards 和 bundle 导出；aeon 更偏算法/toolbox 的广度。",
        combine_pattern_zh="先用 TSDataForge 创建、路由并保存数据资产；当评测层最重要时，再用 aeon 的算法或 benchmark 格式。",
        agent_token_story_zh="aeon 负责算法侧的广度；TSDataForge 帮 agent 理解并推动数据侧资产穿过整个流程。",
    ),
    PackageProfile(
        package_id="stumpy",
        title="STUMPY",
        official_url="https://stumpy.readthedocs.io/",
        one_liner="Efficient matrix-profile tooling for motifs, discords, segmentation, streaming, and related time-series data mining tasks.",
        primary_strengths=("matrix profile", "motif discovery", "discord discovery", "streaming updates"),
        best_for=("subsequence motif search", "anomaly/discord search", "segmentation with matrix profile methods"),
        not_the_main_job=("general dataset taskification", "docs/FAQ/EDA surfaces", "cross-task asset packaging"),
        tsdataforge_difference="TSDataForge is broader and earlier in the workflow. It can point you toward motif or event tasks, but it does not try to be the deepest specialist library for matrix-profile mining.",
        combine_pattern="Use TSDataForge to identify that a sequence or dataset looks motif/event heavy, then use STUMPY when matrix-profile methods are the right downstream analysis.",
        agent_token_story="STUMPY is algorithmically specialized; TSDataForge helps an agent explain why motif-style analysis is being selected in the first place.",
        environment_ids=("python-script", "jupyter-notebook"),
        keywords=("motifs", "discords", "matrix profile", "streaming", "segmentation"),
        one_liner_zh="一个高效的 matrix-profile 工具库，适合 motif、discord、segmentation 和 streaming 等任务。",
        tsdataforge_difference_zh="TSDataForge 位于更前面也更宽。它可以把你路由到 motif 或 event 任务，但不会试图成为最深的 matrix-profile 专家库。",
        combine_pattern_zh="先用 TSDataForge 识别序列或数据集是否偏 motif/event，再在 matrix-profile 方法真正合适时使用 STUMPY。",
        agent_token_story_zh="STUMPY 是算法特化库；TSDataForge 则帮助 agent 先解释为什么要选 motif-style analysis。",
    ),
    PackageProfile(
        package_id="ydata-profiling",
        title="YData Profiling",
        official_url="https://docs.profiling.ydata.ai/",
        one_liner="Automated profiling reports with statistics and visualizations across supported data structures.",
        primary_strengths=("profiling reports", "one-command summaries", "shareable overview reports"),
        best_for=("generic data profiling", "quick descriptive reports", "data dictionaries"),
        not_the_main_job=("time-series task schemas", "control/causal/counterfactual sequence semantics", "spec-driven dataset generation"),
        tsdataforge_difference="TSDataForge is much narrower but more sequence-specific. It profiles time series in a way that routes into tasks, specs, similarity, and agent assets instead of stopping at a generic report.",
        combine_pattern="Use TSDataForge when the data is fundamentally sequential and task semantics matter; use YData Profiling when you need broad tabular or mixed-structure profiling around the same project.",
        agent_token_story="Generic profiling reports are useful, but TSDataForge adds compact task semantics and low-token handoff artifacts around them.",
        environment_ids=("jupyter-notebook", "python-script", "static-docs-site"),
        keywords=("profiling", "reports", "visualization", "metadata", "EDA"),
        one_liner_zh="一个自动生成统计与可视化报告的数据 profiling 库。",
        tsdataforge_difference_zh="TSDataForge 更窄，但更 sequence-specific。它做时间序列 profiling 时，会继续路由到 tasks、specs、similarity 和 agent assets，而不是停在通用报告。",
        combine_pattern_zh="当数据本质上是时序且任务语义很重要时，用 TSDataForge；当项目同时需要更广的表格或混合结构 profiling 时，用 YData Profiling。",
        agent_token_story_zh="通用 profiling 报告当然有用，但 TSDataForge 还会补上 compact task semantics 和低 token handoff artifacts。",
    ),
]


def _translate(item: PackageProfile) -> PackageProfile:
    return PackageProfile(
        package_id=item.package_id,
        title=item.title,
        official_url=item.official_url,
        kind=item.kind,
        one_liner=item.one_liner_zh or item.one_liner,
        primary_strengths=item.primary_strengths,
        best_for=item.best_for,
        not_the_main_job=item.not_the_main_job,
        tsdataforge_difference=item.tsdataforge_difference_zh or item.tsdataforge_difference,
        combine_pattern=item.combine_pattern_zh or item.combine_pattern,
        agent_token_story=item.agent_token_story_zh or item.agent_token_story,
        environment_ids=item.environment_ids,
        keywords=item.keywords,
        one_liner_zh=item.one_liner_zh,
        tsdataforge_difference_zh=item.tsdataforge_difference_zh,
        combine_pattern_zh=item.combine_pattern_zh,
        agent_token_story_zh=item.agent_token_story_zh,
    )


from importlib import import_module


def _version() -> str:
    try:
        tdf = import_module("tsdataforge")
        return str(getattr(tdf, "__version__", "unknown"))
    except Exception:
        return "unknown"


THESIS_EN = (
    "TSDataForge is not trying to own the whole time-series ecosystem. "
    "Its position is narrower and more deliberate: it is the profiling and handoff layer that helps teams understand raw or simulated time-series datasets before modeling or transfer."
)
THESIS_ZH = (
    "TSDataForge 不是想把整个时间序列生态都重做一遍。它的位置更窄也更明确："
    "它是把原始或仿真的序列数据转成可解释、可任务化、可分享、对 agent 友好的资产层。"
)


def competitor_catalog(*, language: str = "en") -> list[PackageProfile]:
    """Return a curated ecosystem map for the public TSDataForge surface.

    The goal is not to rank projects. The goal is to explain where TSDataForge
    fits, what it does not try to replace, and how it combines with well-known
    libraries in a practical workflow.
    """
    if language.startswith("zh"):
        return [_translate(item) for item in _BASE_PROFILES]
    return list(_BASE_PROFILES)


def build_positioning_matrix(*, language: str = "en") -> PositioningMatrix:
    """Build the curated ecosystem-positioning matrix for docs and README generation."""
    return PositioningMatrix(
        version=_version(),
        thesis=THESIS_EN,
        thesis_zh=THESIS_ZH,
        profiles=tuple(competitor_catalog(language=language)),
    )


def render_positioning_markdown(matrix: PositioningMatrix | None = None, *, language: str = "en") -> str:
    """Render the positioning matrix as Markdown for README-like surfaces."""
    mat = matrix or build_positioning_matrix(language=language)
    return mat.to_markdown(language=language)


def save_positioning_matrix(matrix: PositioningMatrix | None, path: str | Path, *, language: str = "en") -> None:
    """Save the positioning matrix as JSON or Markdown."""
    mat = matrix or build_positioning_matrix(language=language)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(mat.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if path.suffix.lower() in {".md", ".markdown"}:
        path.write_text(mat.to_markdown(language=language), encoding="utf-8")
        return
    raise ValueError("Unsupported positioning extension. Use .json or .md")


def _score(item: PackageProfile, query: str) -> int:
    q = query.lower()
    hay = " ".join(
        [
            item.package_id,
            item.title,
            item.one_liner,
            item.tsdataforge_difference,
            item.combine_pattern,
            item.agent_token_story,
            *item.primary_strengths,
            *item.best_for,
            *item.not_the_main_job,
            *item.keywords,
        ]
    ).lower()
    score = 0
    for token in q.replace("/", " ").replace("-", " ").split():
        if token and token in hay:
            score += 1
    return score


def recommend_companions(query: str, *, top_k: int = 3, language: str = "en") -> list[PackageProfile]:
    """Recommend companion libraries to pair with TSDataForge for a given workflow.

    This returns a practical pairing suggestion, not a winner-takes-all ranking.
    The TSDataForge self row is intentionally excluded from the recommendations.
    """
    items = [item for item in competitor_catalog(language=language) if item.kind != "self"]
    scored = sorted(items, key=lambda item: (-_score(item, query), item.title.lower()))
    return scored[: max(1, int(top_k))]


__all__ = [
    "PackageProfile",
    "PositioningMatrix",
    "competitor_catalog",
    "build_positioning_matrix",
    "render_positioning_markdown",
    "save_positioning_matrix",
    "recommend_companions",
]
