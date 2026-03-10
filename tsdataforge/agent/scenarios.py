from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class EnvironmentProfile:
    env_id: str
    title: str
    summary: str
    best_for: tuple[str, ...] = field(default_factory=tuple)
    strengths: tuple[str, ...] = field(default_factory=tuple)
    limitations: tuple[str, ...] = field(default_factory=tuple)
    typical_outputs: tuple[str, ...] = field(default_factory=tuple)
    keywords: tuple[str, ...] = field(default_factory=tuple)
    title_zh: str = ""
    summary_zh: str = ""
    best_for_zh: tuple[str, ...] = field(default_factory=tuple)
    strengths_zh: tuple[str, ...] = field(default_factory=tuple)
    limitations_zh: tuple[str, ...] = field(default_factory=tuple)
    typical_outputs_zh: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioProfile:
    scenario_id: str
    title: str
    one_liner: str
    problem: str
    why_tsdataforge: str
    what_to_do: tuple[str, ...] = field(default_factory=tuple)
    environment_ids: tuple[str, ...] = field(default_factory=tuple)
    api_names: tuple[str, ...] = field(default_factory=tuple)
    example_ids: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)
    audience: tuple[str, ...] = field(default_factory=tuple)
    keywords: tuple[str, ...] = field(default_factory=tuple)
    title_zh: str = ""
    one_liner_zh: str = ""
    problem_zh: str = ""
    why_tsdataforge_zh: str = ""
    what_to_do_zh: tuple[str, ...] = field(default_factory=tuple)
    outputs_zh: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ENVIRONMENTS: tuple[EnvironmentProfile, ...] = (
    EnvironmentProfile(
        env_id="jupyter-notebook",
        title="Jupyter notebook / lab notebook",
        summary="Best for exploratory generation, real-data EDA, debugging one workflow at a time, and teaching users what each object looks like.",
        best_for=("real-data EDA", "quick iteration", "teaching", "report screenshots"),
        strengths=("plots and HTML reports are easy to inspect", "supports step-by-step debugging", "good first environment for new users"),
        limitations=("not ideal for very large batch generation", "easy to accumulate hidden notebook state"),
        typical_outputs=("EDA report", "one series", "small benchmark prototype", "saved dataset asset"),
        keywords=("notebook", "jupyter", "eda", "teaching", "exploration"),
        title_zh="Jupyter notebook / 实验记录环境",
        summary_zh="最适合探索式生成、真实数据 EDA、逐步调试工作流，以及给新用户演示对象和输出长什么样。",
        best_for_zh=("真实数据 EDA", "快速迭代", "教学", "报告截图"),
        strengths_zh=("图和 HTML 报告容易查看", "适合一步一步调试", "最适合作为新用户起步环境"),
        limitations_zh=("不适合非常大的批量生成", "容易积累 notebook 隐状态"),
        typical_outputs_zh=("EDA 报告", "单条序列", "小型 benchmark 原型", "保存好的数据集资产"),
    ),
    EnvironmentProfile(
        env_id="python-script",
        title="Python script / CLI job",
        summary="Best for reproducible pipelines, repeated dataset generation, and packaging a workflow into something other people can rerun.",
        best_for=("dataset generation", "taskification", "example scripts", "scheduled jobs"),
        strengths=("easy to version control", "good for reproducible artifacts", "works well in local and remote environments"),
        limitations=("less interactive than notebooks", "requires more explicit output saving"),
        typical_outputs=("saved base dataset", "task dataset", "docs bundle", "API manifest"),
        keywords=("script", "cli", "pipeline", "reproducible", "batch"),
        title_zh="Python 脚本 / CLI 任务",
        summary_zh="最适合可复现流水线、重复数据集生成，以及把工作流打包成别人也能复跑的东西。",
        best_for_zh=("数据集生成", "任务化", "案例脚本", "定时任务"),
        strengths_zh=("易于版本管理", "适合产出可复现资产", "本地和远程环境都好用"),
        limitations_zh=("交互性不如 notebook", "需要更显式地保存输出"),
        typical_outputs_zh=("保存好的基础数据集", "任务数据集", "docs bundle", "API manifest"),
    ),
    EnvironmentProfile(
        env_id="headless-server",
        title="Headless server / batch node",
        summary="Best for generating large synthetic corpora, repeated benchmark suites, and dataset assets that do not require manual interaction.",
        best_for=("large-scale synthetic generation", "batch benchmarks", "nightly jobs"),
        strengths=("scales repeated generation", "fits CI or scheduled jobs", "easy to archive output bundles"),
        limitations=("visual inspection happens later", "needs explicit report and manifest saving"),
        typical_outputs=("benchmark suite", "task dataset pack", "saved cards and manifests"),
        keywords=("server", "batch", "headless", "benchmark", "scale"),
        title_zh="无头服务器 / 批处理节点",
        summary_zh="最适合大规模合成语料、重复 benchmark suite，以及不需要人工交互的数据资产生成。",
        best_for_zh=("大规模合成生成", "批量 benchmark", "夜间任务"),
        strengths_zh=("适合重复生成", "适合 CI 或定时任务", "方便归档输出 bundle"),
        limitations_zh=("可视化检查通常要后置", "需要显式保存报告和 manifest"),
        typical_outputs_zh=("benchmark suite", "任务数据集包", "保存好的 card 与 manifest"),
    ),
    EnvironmentProfile(
        env_id="ci-pipeline",
        title="CI pipeline / release automation",
        summary="Best for validating examples, regenerating docs, exporting API manifests, and making sure saved assets stay consistent across releases.",
        best_for=("release checks", "docs generation", "example validation", "artifact consistency"),
        strengths=("keeps docs and code in sync", "prevents public-surface drift", "helps absorb external traffic"),
        limitations=("not a discovery environment", "requires disciplined artifact outputs"),
        typical_outputs=("docs site", "API reference", "release bundle", "smoke-tested examples"),
        keywords=("ci", "release", "docs", "validation", "automation"),
        title_zh="CI 流水线 / 发布自动化",
        summary_zh="最适合校验案例、重建文档、导出 API manifest，并确保保存资产在版本间保持一致。",
        best_for_zh=("发布检查", "文档生成", "案例校验", "资产一致性"),
        strengths_zh=("保持文档和代码同步", "防止公共表层漂移", "有助于承接外部流量"),
        limitations_zh=("不是探索环境", "需要纪律化的资产输出"),
        typical_outputs_zh=("docs site", "API reference", "release bundle", "冒烟测试后的案例"),
    ),
    EnvironmentProfile(
        env_id="simulator-or-data-lake",
        title="External simulator / data lake bridge",
        summary="Best when the time series come from another simulator or production data source and TSDataForge should add EDA, taskification, and explanation rather than rebuild the upstream system.",
        best_for=("MuJoCo or ROS rollouts", "production logs", "observability exports", "shared datasets"),
        strengths=("lets TSDataForge stay focused on structure, tasks, and reports", "works with real and synthetic sources"),
        limitations=("quality of results depends on upstream schema quality", "raw signals may need cleaning first"),
        typical_outputs=("wrapped rollout", "EDA report", "taskified dataset", "linked report bundle"),
        keywords=("external", "simulator", "ros", "rollout", "real data", "data lake"),
        title_zh="外部仿真 / 数据湖桥接",
        summary_zh="最适合时序来自其他仿真器或生产数据源，而 TSDataForge 只负责补 EDA、任务化和解释，而不是重建上游系统。",
        best_for_zh=("MuJoCo 或 ROS rollout", "生产日志", "观测系统导出", "共享数据集"),
        strengths_zh=("让 TSDataForge 聚焦结构、任务和报告", "真实和合成来源都能接入"),
        limitations_zh=("结果质量依赖上游 schema 质量", "原始信号可能需要先清洗"),
        typical_outputs_zh=("wrapped rollout", "EDA 报告", "任务化数据集", "linked report bundle"),
    ),
    EnvironmentProfile(
        env_id="llm-agent-workflow",
        title="LLM / agent workflow",
        summary="Best when a coding agent or research assistant needs compact contexts, schema-stable datasets, API manifests, and linked reports instead of raw arrays and long prose.",
        best_for=("agent handoff", "tool-using assistants", "prompt compaction", "structured contexts"),
        strengths=("saves tokens", "keeps semantics explicit", "works well with saved cards and manifests"),
        limitations=("agents still need narrow, stable scopes", "overly large raw assets remain expensive"),
        typical_outputs=("compact context", "artifact card", "task schema", "linked docs bundle"),
        keywords=("agent", "llm", "token", "schema", "context", "manifest"),
        title_zh="LLM / agent 工作流",
        summary_zh="最适合让 coding agent 或研究助手读取 compact context、schema 稳定的数据集、API manifest 和 linked report，而不是原始数组和长篇说明。",
        best_for_zh=("agent 交接", "工具型助手", "prompt 压缩", "结构化上下文"),
        strengths_zh=("节省 token", "让语义保持显式", "很适合与 card 和 manifest 配合"),
        limitations_zh=("agent 仍需要窄而稳定的作用域", "过大的原始资产依然昂贵"),
        typical_outputs_zh=("compact context", "artifact card", "task schema", "linked docs bundle"),
    ),
    EnvironmentProfile(
        env_id="static-docs-site",
        title="Static docs site / shareable bundle",
        summary="Best for public adoption, team onboarding, and sending a report or dataset to someone who should understand it without opening your codebase.",
        best_for=("global docs", "team onboarding", "public examples", "FAQ and support"),
        strengths=("easy to host", "friendly to non-authors", "pairs well with linked EDA reports"),
        limitations=("not an execution environment", "needs regular regeneration"),
        typical_outputs=("docs site", "example pages", "FAQ", "release bundle"),
        keywords=("docs", "site", "bundle", "adoption", "support", "faq"),
        title_zh="静态文档站 / 可分享 bundle",
        summary_zh="最适合公共传播、团队 onboarding，以及把报告或数据集发给别人时让对方无需打开代码库就能理解。",
        best_for_zh=("全球文档", "团队 onboarding", "公开案例", "FAQ 与支持"),
        strengths_zh=("容易托管", "对非作者友好", "很适合和 linked EDA report 配合"),
        limitations_zh=("不是执行环境", "需要定期重建"),
        typical_outputs_zh=("docs site", "案例页", "FAQ", "release bundle"),
    ),
)


_SCENARIOS: tuple[ScenarioProfile, ...] = (
    ScenarioProfile(
        scenario_id="synthetic-benchmark-design",
        title="Design a structure-aware benchmark from scratch",
        one_liner="Generate one reusable base dataset, then derive forecasting, classification, or robustness tasks without losing the connection to the underlying structure.",
        problem="Researchers often need synthetic data with explicit structure, trace-level truth, and repeatable observation conditions.",
        why_tsdataforge="TSDataForge separates latent structure, observation mechanism, and taskification, so the same base asset can support several experiments instead of being thrown away after one task.",
        what_to_do=(
            "Start with generate_series_dataset so the raw structural asset stays reusable.",
            "Run describe_dataset or dataset EDA to inspect coverage and signature balance.",
            "Taskify the same base dataset into forecasting, classification, and robustness views.",
            "Save the dataset with cards, schema, and contexts so collaborators can reuse it.",
        ),
        environment_ids=("python-script", "headless-server", "ci-pipeline"),
        api_names=("generate_series_dataset", "describe_dataset", "generate_dataset_eda_report", "taskify_dataset", "SeriesDataset"),
        example_ids=("taskify_forecasting", "classification_benchmark", "robust_observation_variants"),
        outputs=("base dataset", "task datasets", "dataset card", "EDA report"),
        audience=("ML researchers", "benchmark maintainers", "PhD students"),
        keywords=("benchmark", "synthetic", "forecasting", "classification", "robustness"),
        title_zh="从零设计结构感知 benchmark",
        one_liner_zh="先生成一份可复用基础数据集，再派生 forecasting、classification 或 robustness 任务，同时保持与底层结构的联系。",
        problem_zh="研究者经常需要带显式结构、trace 真值和可复现实验观测条件的合成数据。",
        why_tsdataforge_zh="TSDataForge 把 latent structure、observation mechanism 和 taskification 分开，因此同一份基础资产可以支持多类实验，而不是做完一个任务就丢掉。",
        what_to_do_zh=(
            "先用 generate_series_dataset，保证原始结构资产可复用。",
            "再用 describe_dataset 或 dataset EDA 检查覆盖率和结构平衡。",
            "把同一份基础数据集 taskify 成 forecasting、classification 和 robustness 视图。",
            "保存带 card、schema 和 context 的数据集，方便协作者复用。",
        ),
        outputs_zh=("基础数据集", "任务数据集", "dataset card", "EDA 报告"),
    ),
    ScenarioProfile(
        scenario_id="real-data-profile-and-taskify",
        title="Profile a real dataset before deciding the task",
        one_liner="Turn unknown raw time series into a report, a provisional structure label set, and a shortlist of sensible tasks.",
        problem="Many teams jump straight into modeling before they know whether their data is forecasting-like, event-like, control-like, or causal-like.",
        why_tsdataforge="The library gives you describe_series, describe_dataset, human-readable explanations, and linked EDA reports before you commit to a modeling task.",
        what_to_do=(
            "Wrap the raw arrays or external rollouts into SeriesDataset or GeneratedSeries objects.",
            "Generate EDA reports and inspect the linked next steps.",
            "Choose the nearest task view instead of forcing every dataset into forecasting.",
            "Save the results as a linked bundle for teammates.",
        ),
        environment_ids=("jupyter-notebook", "python-script", "static-docs-site"),
        api_names=("describe_series", "describe_dataset", "generate_eda_report", "generate_dataset_eda_report", "SeriesDataset"),
        example_ids=("real_series_eda", "dataset_eda", "real_dataset_taskify", "linked_eda_bundle"),
        outputs=("EDA report", "recommended tasks", "structure tags", "shared bundle"),
        audience=("applied researchers", "data scientists", "lab engineers"),
        keywords=("real data", "eda", "profiling", "task selection", "report"),
        title_zh="在决定任务前先画像真实数据集",
        one_liner_zh="把未知原始时序转成报告、初步结构标签以及一份合理任务 shortlist。",
        problem_zh="很多团队在搞清楚数据更像 forecasting、event、control 还是 causal 之前，就直接开始建模。",
        why_tsdataforge_zh="这个库在你真正决定任务之前，就给你 describe_series、describe_dataset、解释性文字和 linked EDA 报告。",
        what_to_do_zh=(
            "把原始数组或外部 rollout 包装成 SeriesDataset 或 GeneratedSeries。",
            "生成 EDA 报告并查看里面联动的下一步建议。",
            "选择最接近的数据任务视图，而不是把所有数据都强行做 forecasting。",
            "把结果保存成 linked bundle 方便团队共享。",
        ),
        outputs_zh=("EDA 报告", "推荐任务", "结构标签", "共享 bundle"),
    ),
    ScenarioProfile(
        scenario_id="one-base-dataset-many-tasks",
        title="Reuse one base dataset across many tasks",
        one_liner="Keep the raw time-series asset stable while deriving many task-specific views with explicit schema.",
        problem="A common failure mode is to generate a bespoke dataset per task and lose consistency across experiments.",
        why_tsdataforge="SeriesDataset and TaskDataset separate the reusable asset layer from the task protocol layer, so multiple tasks remain comparable.",
        what_to_do=(
            "Generate or import a base SeriesDataset once.",
            "Call taskify repeatedly for forecasting, classification, system identification, SSL, or causal response.",
            "Inspect TaskDataset.schema to keep semantics explicit.",
            "Save each task view next to the base asset.",
        ),
        environment_ids=("python-script", "jupyter-notebook", "llm-agent-workflow"),
        api_names=("SeriesDataset", "TaskDataset", "taskify_dataset", "TaskSpec", "build_task_context"),
        example_ids=("taskify_forecasting", "real_dataset_taskify", "system_identification", "causal_response"),
        outputs=("multiple task datasets", "shared schema", "cards and contexts"),
        audience=("benchmark builders", "platform engineers", "multitask teams"),
        keywords=("reuse", "taskify", "multitask", "schema", "forecasting", "classification"),
        title_zh="一份基础数据集服务多种任务",
        one_liner_zh="让原始时序资产保持稳定，同时派生出多种带显式 schema 的任务视图。",
        problem_zh="一个常见失败模式是每个任务单独生成一份定制数据集，导致实验之间失去一致性。",
        why_tsdataforge_zh="SeriesDataset 和 TaskDataset 把可复用资产层和任务协议层分开，使多任务之间仍然可比较。",
        what_to_do_zh=(
            "先生成或导入一份基础 SeriesDataset。",
            "然后反复调用 taskify 去派生 forecasting、classification、system identification、SSL 或 causal response。",
            "查看 TaskDataset.schema，保证语义显式。",
            "把各个任务视图和基础资产一起保存。",
        ),
        outputs_zh=("多种任务数据集", "共享 schema", "cards 与 contexts"),
    ),
    ScenarioProfile(
        scenario_id="control-rollout-analysis",
        title="Analyse control-like or rollout data without rebuilding the simulator",
        one_liner="Bring external rollouts or control-flavoured series into TSDataForge, add structure-aware reports, then derive system-identification or intervention tasks.",
        problem="Control and robotics teams already have simulators; they usually need interpretation, taskification, and explainable assets rather than another simulator.",
        why_tsdataforge="TSDataForge can wrap external rollouts, describe their structure, and convert them into datasets for system identification, intervention detection, policy evaluation, or counterfactual analysis.",
        what_to_do=(
            "Use wrap_external_series or SeriesDataset.from_arrays to ingest the signals.",
            "Run linked EDA reports to identify control-like patterns, events, and observation issues.",
            "Taskify the dataset into system identification or intervention-related tasks.",
            "Ship the resulting dataset with schema and linked reports.",
        ),
        environment_ids=("simulator-or-data-lake", "jupyter-notebook", "python-script"),
        api_names=("wrap_external_series", "generate_eda_report", "taskify_dataset", "LinearStateSpace", "InterventionSpec"),
        example_ids=("external_rollout_wrap", "system_identification", "event_control_detection", "intervention_detection"),
        outputs=("wrapped rollout", "EDA report", "system ID dataset", "intervention dataset"),
        audience=("control researchers", "robotics teams", "simulation engineers"),
        keywords=("control", "rollout", "system identification", "robotics", "external simulator"),
        title_zh="不重造仿真器，直接分析控制类 rollout 数据",
        one_liner_zh="把外部 rollout 或控制风格时序接入 TSDataForge，补结构感知报告，再派生 system identification 或 intervention 任务。",
        problem_zh="控制和机器人团队通常已经有仿真器；他们更需要解释、任务化和可共享资产，而不是另一个仿真器。",
        why_tsdataforge_zh="TSDataForge 可以包装外部 rollout、描述其结构，并把它们转换成 system identification、intervention detection、policy evaluation 或 counterfactual analysis 数据集。",
        what_to_do_zh=(
            "用 wrap_external_series 或 SeriesDataset.from_arrays 接入信号。",
            "通过 linked EDA report 识别 control-like 模式、事件和观测问题。",
            "把数据 taskify 成 system identification 或 intervention 相关任务。",
            "用 schema 和 linked report 一起发布结果数据集。",
        ),
        outputs_zh=("wrapped rollout", "EDA 报告", "system ID 数据集", "intervention 数据集"),
    ),
    ScenarioProfile(
        scenario_id="causal-and-counterfactual-evaluation",
        title="Create causal or counterfactual evaluation assets",
        one_liner="Generate or inspect treatment, response, graph, and counterfactual structure so causal pipelines have explicit targets and narrative evidence.",
        problem="Causal time-series work often lacks clean task interfaces, visible assumptions, or explainable outputs for collaborators.",
        why_tsdataforge="The package combines causal generators, intervention and policy abstractions, taskification, and EDA/reporting so causal workflows can stay structured and auditable.",
        what_to_do=(
            "Use causal generators or imported datasets that expose treatment or intervention semantics.",
            "Taskify into causal_response, causal_discovery, counterfactual_response, or related views.",
            "Compare factual and counterfactual trajectories with linked reports.",
            "Save schema, cards, and contexts so the causal assumptions remain explicit.",
        ),
        environment_ids=("python-script", "jupyter-notebook", "headless-server"),
        api_names=("CausalVARX", "CausalTreatmentOutcome", "InterventionSpec", "generate_counterfactual_pair", "taskify_dataset"),
        example_ids=("causal_response", "policy_counterfactual", "policy_value_estimation"),
        outputs=("causal dataset", "counterfactual pair", "task schema", "report bundle"),
        audience=("causal ML researchers", "policy evaluation teams", "applied scientists"),
        keywords=("causal", "counterfactual", "intervention", "policy", "graph", "ite"),
        title_zh="生成因果或反事实评测资产",
        one_liner_zh="生成或分析 treatment、response、graph 和 counterfactual 结构，让因果流水线拥有显式目标和可解释证据。",
        problem_zh="因果时间序列工作常常缺少干净的任务接口、可见的假设，以及能给协作者解释的输出。",
        why_tsdataforge_zh="这个包把 causal generators、intervention / policy 抽象、任务化和 EDA / 报告结合起来，让因果工作流更结构化、更可审计。",
        what_to_do_zh=(
            "使用 causal 生成器，或者导入带 treatment / intervention 语义的数据集。",
            "taskify 成 causal_response、causal_discovery、counterfactual_response 等视图。",
            "通过 linked report 比较 factual 和 counterfactual 轨迹。",
            "保存 schema、card 和 context，让因果假设保持显式。",
        ),
        outputs_zh=("因果数据集", "反事实对照对", "任务 schema", "报告 bundle"),
    ),
    ScenarioProfile(
        scenario_id="agent-grounded-assets",
        title="Hand off time-series assets to an agent without wasting tokens",
        one_liner="Turn datasets and reports into compact contexts, cards, manifests, and linked docs so an agent sees the semantics before it sees the raw arrays.",
        problem="Agent workflows become expensive and brittle when they receive raw arrays, long README files, and ambiguous task semantics.",
        why_tsdataforge="The library already knows the dataset role, task schema, and EDA findings; it can compress those into a token-efficient public surface.",
        what_to_do=(
            "Build series, dataset, or task contexts with the smallest budget that still preserves semantics.",
            "Save artifact cards next to the datasets.",
            "Export API manifests and linked docs so the agent can route itself.",
            "Use the linked EDA bundle as the human-readable companion layer.",
        ),
        environment_ids=("llm-agent-workflow", "python-script", "static-docs-site"),
        api_names=("build_agent_context", "build_dataset_context", "build_task_context", "build_task_dataset_card", "build_api_reference"),
        example_ids=("agent_context_pack", "dataset_cards", "api_reference_overview", "linked_eda_bundle"),
        outputs=("compact context", "artifact card", "API manifest", "linked bundle"),
        audience=("agent engineers", "MLOps teams", "platform builders"),
        keywords=("agent", "token", "context", "card", "manifest", "schema"),
        title_zh="把时序资产交给 agent，同时不浪费 token",
        one_liner_zh="把数据集和报告变成 compact context、card、manifest 和 linked docs，让 agent 先看到语义，再看到原始数组。",
        problem_zh="当 agent 工作流收到原始数组、长 README 和含糊任务语义时，会变得昂贵又脆弱。",
        why_tsdataforge_zh="库本身已经知道数据集角色、任务 schema 和 EDA findings，因此可以把这些压成更省 token 的公共表层。",
        what_to_do_zh=(
            "用尽可能小但仍保持语义的预算构建 series、dataset 或 task context。",
            "把 artifact card 保存到数据集旁边。",
            "导出 API manifest 和 linked docs，让 agent 能自我路由。",
            "把 linked EDA bundle 作为给人看的配套层。",
        ),
        outputs_zh=("compact context", "artifact card", "API manifest", "linked bundle"),
    ),
    ScenarioProfile(
        scenario_id="teaching-and-onboarding",
        title="Teach a lab or onboard new users fast",
        one_liner="Use the package as a teaching scaffold: short quickstart, runnable examples, then real-data EDA and taskification.",
        problem="New students or collaborators often drown in module names before they see one successful workflow end to end.",
        why_tsdataforge="TSDataForge already has the ingredients for a learning ladder: one-series generation, EDA reports, reusable datasets, taskification, and static docs.",
        what_to_do=(
            "Start from the quickstart and one runnable example.",
            "Move to real-data EDA so the user learns to read the data before modelling.",
            "Show how one base dataset becomes several tasks.",
            "Publish the docs bundle so the same path survives after the workshop.",
        ),
        environment_ids=("jupyter-notebook", "static-docs-site", "python-script"),
        api_names=("generate_series", "generate_eda_report", "generate_series_dataset", "taskify_dataset", "generate_docs_site"),
        example_ids=("quickstart", "real_series_eda", "taskify_forecasting", "docs_site_generation"),
        outputs=("teaching notebook", "saved example outputs", "docs bundle"),
        audience=("teachers", "PIs", "lab managers", "new users"),
        keywords=("teaching", "onboarding", "quickstart", "lab", "tutorial"),
        title_zh="快速教学和实验室 onboarding",
        one_liner_zh="把这个包当成教学脚手架：短 quickstart、可运行案例，再进入真实数据 EDA 和任务化。",
        problem_zh="新学生或协作者常常在看到一条完整成功工作流之前，就先淹没在模块名里。",
        why_tsdataforge_zh="TSDataForge 已经具备学习阶梯所需的元素：单序列生成、EDA 报告、可复用数据集、任务化和静态文档。",
        what_to_do_zh=(
            "先从 quickstart 和一个可运行案例开始。",
            "再进入真实数据 EDA，让用户先学会读数据，再学建模。",
            "展示一份基础数据集如何派生出多个任务。",
            "发布 docs bundle，让这条路径在 workshop 之后仍然存在。",
        ),
        outputs_zh=("教学 notebook", "保存好的案例输出", "docs bundle"),
    ),
    ScenarioProfile(
        scenario_id="docs-launch-and-adoption",
        title="Package the library for global adoption",
        one_liner="Make the public surface explain itself: an English-first docs site, bilingual switch, scenario framing, API rationale, FAQ, and linked EDA bundles.",
        problem="A capable research library can still fail to spread if users cannot understand where to start, what each API is for, or whether it fits their environment.",
        why_tsdataforge="This package already contains docs generation, FAQ, API manifests, example catalogs, and linked EDA bundles, so adoption assets can be built from the same source of truth as the code.",
        what_to_do=(
            "Generate the docs site and inspect the landing, scenarios, examples, API, and FAQ pages together.",
            "Publish a few linked EDA bundles as concrete entry points.",
            "Keep API manifests and example catalogs in the release assets.",
            "Use bilingual docs when the user base is global but collaboration crosses English and Chinese.",
        ),
        environment_ids=("static-docs-site", "ci-pipeline", "llm-agent-workflow"),
        api_names=("generate_docs_site", "build_api_reference", "example_catalog", "build_eda_resource_hub", "generate_linked_eda_bundle"),
        example_ids=("docs_site_generation", "api_reference_overview", "linked_eda_bundle"),
        outputs=("docs site", "API manifest", "example catalog", "shareable report bundles"),
        audience=("maintainers", "open-source teams", "research leads"),
        keywords=("docs", "adoption", "landing page", "faq", "examples", "global"),
        title_zh="为全球传播打包这个库",
        one_liner_zh="让公共表层自己会解释：英文优先 docs site、双语切换、场景 framing、API rationale、FAQ 和 linked EDA bundle。",
        problem_zh="一个功能很强的研究库仍可能传播失败，只因为用户搞不清楚从哪开始、每个 API 是干什么的、是否适合自己的环境。",
        why_tsdataforge_zh="这个包已经内置 docs 生成、FAQ、API manifest、案例目录和 linked EDA bundle，因此采用资产可以和代码共享同一个事实来源。",
        what_to_do_zh=(
            "生成 docs site，并把 landing、scenarios、examples、API 和 FAQ 一起检查一遍。",
            "发布几份 linked EDA bundle 作为具体入口。",
            "把 API manifest 和 example catalog 一起放进 release 资产。",
            "当用户群体是全球研究者、而协作又跨英语与中文时，用双语文档承接。",
        ),
        outputs_zh=("docs site", "API manifest", "案例目录", "可分享报告 bundle"),
    ),
)


def _norm_tokens(*values: str | tuple[str, ...]) -> set[str]:
    text_parts: list[str] = []
    for value in values:
        if isinstance(value, tuple):
            text_parts.extend(str(item) for item in value)
        else:
            text_parts.append(str(value))
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(" ".join(text_parts))}



def environment_catalog(*, language: str = "en") -> list[EnvironmentProfile]:
    if language.startswith("zh"):
        return [
            replace(
                item,
                title=item.title_zh or item.title,
                summary=item.summary_zh or item.summary,
                best_for=item.best_for_zh or item.best_for,
                strengths=item.strengths_zh or item.strengths,
                limitations=item.limitations_zh or item.limitations,
                typical_outputs=item.typical_outputs_zh or item.typical_outputs,
            )
            for item in _ENVIRONMENTS
        ]
    return list(_ENVIRONMENTS)



def scenario_catalog(*, language: str = "en") -> list[ScenarioProfile]:
    if language.startswith("zh"):
        return [
            replace(
                item,
                title=item.title_zh or item.title,
                one_liner=item.one_liner_zh or item.one_liner,
                problem=item.problem_zh or item.problem,
                why_tsdataforge=item.why_tsdataforge_zh or item.why_tsdataforge,
                what_to_do=item.what_to_do_zh or item.what_to_do,
                outputs=item.outputs_zh or item.outputs,
            )
            for item in _SCENARIOS
        ]
    return list(_SCENARIOS)



def recommend_scenarios(query: str, *, top_k: int = 4, language: str = "en") -> list[ScenarioProfile]:
    catalog = scenario_catalog(language=language)
    q = _norm_tokens(query)
    scored: list[tuple[float, str, ScenarioProfile]] = []
    for item in catalog:
        tokens = _norm_tokens(item.title, item.one_liner, item.problem, item.why_tsdataforge, item.keywords, item.api_names, item.example_ids, item.environment_ids)
        score = float(len(q & tokens)) + 0.25 * len(q & set(item.keywords))
        scored.append((score, item.scenario_id, item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in scored[:max(1, top_k)]]



def recommend_environments(query: str, *, top_k: int = 4, language: str = "en") -> list[EnvironmentProfile]:
    catalog = environment_catalog(language=language)
    q = _norm_tokens(query)
    scored: list[tuple[float, str, EnvironmentProfile]] = []
    for item in catalog:
        tokens = _norm_tokens(item.title, item.summary, item.best_for, item.strengths, item.limitations, item.keywords, item.typical_outputs)
        score = float(len(q & tokens))
        scored.append((score, item.env_id, item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in scored[:max(1, top_k)]]


__all__ = [
    "EnvironmentProfile",
    "ScenarioProfile",
    "environment_catalog",
    "scenario_catalog",
    "recommend_scenarios",
    "recommend_environments",
]



_SCENARIOS = _SCENARIOS + (
    ScenarioProfile(
        scenario_id="live-market-and-attention-analysis",
        title="Analyze live market and public-attention series with the same workflow",
        one_liner="Fetch GitHub star histories, crypto prices, or gold/oil series, then describe and compare them with one explainable interface.",
        problem="Many popular public datasets are easy to chart but hard to turn into rigorous, reproducible time-series case studies that teach a reusable method.",
        why_tsdataforge="TSDataForge can fetch the live series, route them through EDA, align them into a shared comparison space, and export a report bundle that both humans and agents can use.",
        what_to_do=(
            "Choose a public signal family such as GitHub attention, crypto, or commodities.",
            "Fetch the series and generate one EDA report before doing similarity analysis.",
            "Use pairwise_similarity for panel-wide comparison or find_top_matches for one reference series.",
            "Publish the report, bundle, and nearest example so the case study also acts as onboarding material.",
        ),
        environment_ids=("jupyter-notebook", "python-script", "static-docs-site", "llm-agent-workflow"),
        api_names=("fetch_github_stars_series", "fetch_coingecko_market_chart", "fetch_fred_series", "generate_eda_report", "pairwise_similarity", "find_top_matches"),
        example_ids=("openclaw_stars_similarity", "github_stars_pairwise_panel", "btc_gold_oil_similarity"),
        outputs=("EDA report", "similarity matrix", "ranked matches", "shareable bundle"),
        audience=("research communicators", "applied researchers", "maintainers", "community builders"),
        keywords=("live data", "github stars", "bitcoin", "gold", "oil", "similarity", "public attention"),
        title_zh="用同一工作流分析实时市场与公共关注度序列",
        one_liner_zh="抓取 GitHub star 历史、加密货币价格或黄金/原油序列，再用同一套可解释接口做描述和比较。",
        problem_zh="很多公共热点数据很容易画图，但很难被转成严格、可复现、还能教学的时间序列案例。",
        why_tsdataforge_zh="TSDataForge 可以抓取实时序列、走 EDA、对齐到共享比较空间，再导出同时适合人和 agent 使用的报告 bundle。",
        what_to_do_zh=(
            "先选一类公共信号：GitHub 关注度、加密货币，或大宗商品。",
            "先抓取序列并生成一份 EDA 报告，再做相似性分析。",
            "面板比较用 pairwise_similarity；单参考匹配用 find_top_matches。",
            "发布报告、bundle 和最近的案例，让这个案例本身也能承担 onboarding 作用。",
        ),
        outputs_zh=("EDA 报告", "相似性矩阵", "排序匹配", "可分享 bundle"),
    ),
)
