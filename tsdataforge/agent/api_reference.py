from __future__ import annotations

import inspect
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_CJK_RE = re.compile(r"[㐀-鿿]")


@dataclass(frozen=True)
class APISymbol:
    name: str
    kind: str
    module: str
    signature: str
    summary: str
    when_to_use: str = ""
    why_exists: str = ""
    works_in: tuple[str, ...] = field(default_factory=tuple)
    scenario_ids: tuple[str, ...] = field(default_factory=tuple)
    summary_zh: str = ""
    when_to_use_zh: str = ""
    why_exists_zh: str = ""
    related: tuple[str, ...] = field(default_factory=tuple)
    example_ids: tuple[str, ...] = field(default_factory=tuple)
    group: str = "core"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class APICategory:
    category_id: str
    title: str
    summary: str
    symbols: tuple[APISymbol, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category_id": self.category_id,
            "title": self.title,
            "summary": self.summary,
            "symbols": [item.to_dict() for item in self.symbols],
        }


@dataclass(frozen=True)
class APIReference:
    version: str
    categories: tuple[APICategory, ...]
    n_symbols: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "n_symbols": self.n_symbols,
            "categories": [item.to_dict() for item in self.categories],
        }

    def to_markdown(self, *, include_examples: bool = True) -> str:
        from .scenarios import environment_catalog, scenario_catalog

        env_titles = {item.env_id: item.title for item in environment_catalog(language="en")}
        scenario_titles = {item.scenario_id: item.title for item in scenario_catalog(language="en")}
        lines = [
            f"# TSDataForge API Reference v{self.version}",
            "",
            "TSDataForge is a Python library for turning time-series structure into reusable assets, task-specific datasets, real-data EDA reports, and agent-friendly contexts.",
            "",
            f"Public symbols: **{self.n_symbols}**",
            "",
            "## How to read this reference",
            "",
            "1. Start from the category that matches your workflow.",
            "2. For each symbol, check **What it does**, **Why it exists**, and **Works well in** before reading all related APIs.",
            "3. Use the related examples as the fastest path to a working pattern.",
            "",
        ]
        for cat in self.categories:
            lines.append(f"## {cat.title}")
            lines.append("")
            lines.append(cat.summary)
            lines.append("")
            for sym in cat.symbols:
                lines.append(f"### `{sym.name}`")
                lines.append("")
                lines.append(f"- Kind: {sym.kind}")
                lines.append(f"- Module: `{sym.module}`")
                lines.append(f"- Signature: `{sym.signature}`")
                lines.append(f"- What it does: {sym.summary}")
                if sym.why_exists:
                    lines.append(f"- Why it exists: {sym.why_exists}")
                if sym.when_to_use:
                    lines.append(f"- Use it when: {sym.when_to_use}")
                if sym.works_in:
                    env_labels = ', '.join(f'`{env_titles.get(item, item)}`' for item in sym.works_in)
                    lines.append(f"- Works well in: {env_labels}")
                if sym.scenario_ids:
                    scenario_labels = ', '.join(f'`{scenario_titles.get(item, item)}`' for item in sym.scenario_ids)
                    lines.append(f"- Common scenarios: {scenario_labels}")
                if sym.related:
                    lines.append(f"- Related APIs: {', '.join(f'`{item}`' for item in sym.related)}")
                if include_examples and sym.example_ids:
                    lines.append(f"- Related examples: {', '.join(f'`{item}`' for item in sym.example_ids)}")
                lines.append("")
        return "\n".join(lines).strip() + "\n"


_GROUPS: dict[str, dict[str, str]] = {
    "workflow": {
        "title": "Workflow entry points",
        "summary": "The most common paths begin here: generate a series, generate a base dataset, taskify it, or generate a task dataset directly.",
    },
    "containers": {
        "title": "Core containers and specs",
        "summary": "Spec, trace, series, dataset, and task-dataset objects live here.",
    },
    "analysis": {
        "title": "Description, explanation, and EDA",
        "summary": "Describe first, model second. This group covers tags, explanations, and HTML reports.",
    },
    "agent": {
        "title": "Agent, docs, and launch surface",
        "summary": "Low-token interfaces for prompts, agents, artifact cards, and static docs sites.",
    },
    "causal_control": {
        "title": "Control, causal, and counterfactual",
        "summary": "Interfaces for inputs, interventions, policies, counterfactuals, and causal structure.",
    },
    "operators": {
        "title": "Composition and observation layer",
        "summary": "Use these to combine, modulate, warp, and observe latent structure.",
    },
    "primitives": {
        "title": "Primitive generators and dynamics",
        "summary": "Trend, seasonality, noise, events, state space, and control dynamics.",
    },
}


_OVERRIDES: dict[str, dict[str, Any]] = {

    "load_asset": {
        "group": "workflow",
        "summary": "Load a path or raw arrays into a reusable SeriesDataset or TaskDataset asset.",
        "when_to_use": "Use this when you want one obvious loader for .npy, .npz, .csv, .txt, .json, or raw arrays.",
        "related": ("report", "handoff", "taskify"),
        "example_ids": ("csv_to_report", "npy_to_handoff_bundle"),
    },
    "report": {
        "group": "analysis",
        "summary": "Generate the first human-readable EDA report from a series or base dataset.",
        "when_to_use": "Use this when you want the shortest path from asset to HTML explanation.",
        "related": ("load_asset", "handoff", "generate_eda_report", "generate_dataset_eda_report"),
        "example_ids": ("real_series_eda", "dataset_eda"),
    },
    "handoff": {
        "group": "agent",
        "summary": "Build the shortest report + card + context + schema + next-action bundle from an asset.",
        "when_to_use": "Use this when you want the package to feel like a dataset handoff tool, with a tiny first-entry contract for agents and a believable recommended next step.",
        "related": ("load_asset", "report", "build_dataset_handoff_bundle"),
        "example_ids": ("dataset_handoff_bundle", "npy_to_handoff_bundle"),
    },
    "taskify": {
        "group": "workflow",
        "summary": "Taskify a base dataset after you have already understood the asset.",
        "when_to_use": "Use this when the report is done and you are finally ready to choose forecasting, anomaly, causal, or control views.",
        "related": ("load_asset", "taskify_dataset", "SeriesDataset"),
        "example_ids": ("real_dataset_taskify", "taskify_forecasting"),
    },
    "demo": {
        "group": "agent",
        "summary": "Generate the built-in public demo bundle for first-success onboarding, with real public demos such as ECG, macro, climate, and sunspots plus optional reality-shaped synthetic showcases.",
        "when_to_use": "Use this for GitHub screenshots, workshops, smoke tests, and the fastest copy-paste success path. Use the `scenario` argument to switch between real public demos and synthetic showcases.",
        "related": ("handoff", "build_dataset_handoff_bundle"),
        "example_ids": ("quickstart",),
    },
    "PublicSurface": {
        "group": "agent",
        "summary": "Structured description of the five public entry points new users should learn first.",
        "when_to_use": "Use this when you want docs or agents to consume the curated public surface instead of the full research API.",
        "related": ("public_surface", "render_public_surface_markdown"),
    },
    "PublicEntrypoint": {
        "group": "agent",
        "summary": "One public entry point in the curated TSDataForge surface.",
        "when_to_use": "Use this for docs generation, onboarding, and public API communication.",
        "related": ("PublicSurface",),
    },
    "public_surface": {
        "group": "agent",
        "summary": "Return the five entry points that define the public-facing TSDataForge product surface.",
        "when_to_use": "Use this when README, docs, or agents should stay aligned to the same small API set.",
        "related": ("PublicSurface", "render_public_surface_markdown", "save_public_surface"),
    },
    "render_public_surface_markdown": {
        "group": "agent",
        "summary": "Render the curated public surface into Markdown.",
        "when_to_use": "Use this when README-like output or low-token docs need the same surface manifest.",
        "related": ("public_surface", "save_public_surface"),
    },
    "save_public_surface": {
        "group": "agent",
        "summary": "Save the curated public surface as Markdown or JSON.",
        "when_to_use": "Use this when docs bundles or CI need a stable public-surface artifact.",
        "related": ("public_surface",),
    },
    "ActionPlanItem": {
        "group": "agent",
        "summary": "One structured next-step item inside a handoff bundle action plan.",
        "when_to_use": "Use this when agents or reviewers need to separate already-done work from the single recommended next step.",
        "related": ("DatasetHandoffBundle", "HandoffIndex"),
    },
    "ArtifactSchema": {
        "group": "agent",
        "summary": "Schema metadata for one handoff artifact format.",
        "when_to_use": "Use this when consumers need a formal contract for context, card, or bundle JSON.",
        "related": ("ArtifactSchemaCatalog", "build_artifact_schemas"),
    },
    "ArtifactSchemaCatalog": {
        "group": "agent",
        "summary": "Collection of JSON schemas for the main handoff artifacts.",
        "when_to_use": "Use this when you want a schema-first contract for agents or external tooling.",
        "related": ("build_artifact_schemas", "save_artifact_schemas"),
    },
    "build_artifact_schemas": {
        "group": "agent",
        "summary": "Build JSON Schema definitions for dataset_context, dataset_card, handoff_index_min, handoff_index, action_plan, and handoff_bundle.",
        "when_to_use": "Use this when external tools should validate or consume TSDataForge artifacts without reading the Python code.",
        "related": ("save_artifact_schemas",),
    },
    "save_artifact_schemas": {
        "group": "agent",
        "summary": "Write the artifact schema set to disk as `.schema.json` files plus a catalog.",
        "when_to_use": "Use this when a handoff bundle or docs site should ship schema-first contracts alongside the artifacts.",
        "related": ("build_artifact_schemas",),
    },
    "build_demo_dataset": {
        "group": "workflow",
        "summary": "Build one of the built-in showcase datasets, including real public ECG, macro, climate, and sunspot demos plus optional synthetic bundles.",
        "when_to_use": "Use this when you need a credible demo asset for README screenshots, workshops, or public docs.",
        "related": ("demo", "demo_scenario_catalog"),
    },
    "demo_scenario_catalog": {
        "group": "agent",
        "summary": "List the built-in demo scenarios and their narratives, with real public demos surfaced first.",
        "when_to_use": "Use this when docs, starter kits, or CLI help should explain the available flagship demos.",
        "related": ("build_demo_dataset", "demo"),
    },
    "generate_series": {
        "group": "workflow",
        "summary": "把 spec 或 components 编译成一条 GeneratedSeries。",
        "when_to_use": "要快速验证一个结构、查看 trace，或给 EDA / 对照实验准备单条样本时。",
        "related": ("compose_series", "SeriesSpec", "ObservationSpec", "generate_eda_report"),
        "example_ids": ("quickstart_univariate", "real_series_eda"),
    },
    "compose_series": {
        "group": "workflow",
        "summary": "用简洁 additive 组件列表生成一条序列。",
        "when_to_use": "你不想显式写 SeriesSpec，只想快速组合几个 primitives 时。",
        "related": ("generate_series", "Add"),
        "example_ids": ("quickstart_univariate",),
    },
    "generate_series_dataset": {
        "group": "workflow",
        "summary": "生成基础 SeriesDataset，而不是直接生成某个任务的 X/y。",
        "when_to_use": "你希望同一份基础数据复用到 forecasting、classification、causal 等多个任务时。",
        "related": ("SeriesDataset", "taskify_dataset", "generate_dataset"),
        "example_ids": ("taskify_forecasting", "classification_benchmark"),
    },
    "taskify_dataset": {
        "group": "workflow",
        "summary": "把 SeriesDataset 转成任务专有的 TaskDataset。",
        "when_to_use": "你已经有基础数据，想派生 forecasting、classification、causal、control、SSL 任务时。",
        "related": ("SeriesDataset", "TaskDataset", "generate_series_dataset"),
        "example_ids": ("taskify_forecasting", "real_dataset_taskify"),
    },
    "generate_dataset": {
        "group": "workflow",
        "summary": "一步到位直接生成任务专有数据集。",
        "when_to_use": "你已经知道任务类型，只想最快得到可训练的 X/y 时。",
        "related": ("generate_series_dataset", "taskify_dataset", "TaskDataset"),
        "example_ids": ("taskify_forecasting", "causal_response", "system_identification"),
    },
    "Compiler": {
        "group": "workflow",
        "summary": "执行 SeriesSpec 并返回 GeneratedSeries 的底层编译器。",
        "when_to_use": "需要显式控制 seed、逐步调试 spec 编译行为，或写自定义流水线时。",
        "related": ("SeriesSpec", "GeneratedSeries", "generate_series"),
        "example_ids": ("quickstart_univariate",),
    },
    "SeriesSpec": {
        "group": "containers",
        "summary": "可序列化的结构规范，描述 latent 组合与观测机制。",
        "when_to_use": "需要可复现、可保存、可比较的时间序列结构程序时。",
        "related": ("ObservationSpec", "Compiler", "GeneratedSeries"),
        "example_ids": ("quickstart_univariate", "policy_counterfactual"),
    },
    "ObservationSpec": {
        "group": "containers",
        "summary": "描述采样、不规则采样、缺失、测量噪声和变换。",
        "when_to_use": "你想把相同 latent structure 放到不同观测机制下做鲁棒性测试时。",
        "related": ("SeriesSpec", "IrregularSampling", "MeasurementNoise"),
        "example_ids": ("robust_observation_variants",),
    },
    "GeneratedSeries": {
        "group": "containers",
        "summary": "单条序列容器，包含 time、values、spec 与 trace。",
        "when_to_use": "需要查看单样本的结构、状态、干预痕迹或保存 artifact 时。",
        "related": ("SeriesTrace", "SeriesDataset", "generate_series"),
        "example_ids": ("quickstart_univariate", "real_series_eda"),
    },
    "SeriesTrace": {
        "group": "containers",
        "summary": "保存 latent、observed、states、masks、contributions 等真值。",
        "when_to_use": "需要 change point、event、adjacency、ITE、counterfactual 这类真值时。",
        "related": ("GeneratedSeries", "taskify_dataset"),
        "example_ids": ("event_control_detection", "policy_counterfactual"),
    },
    "SeriesDataset": {
        "group": "containers",
        "summary": "基础数据集层，保存多条原始序列和元信息。",
        "when_to_use": "你要把数据当成可复用资产，而不是一次性脚本输出时。",
        "related": ("TaskDataset", "taskify_dataset", "build_dataset_context"),
        "example_ids": ("taskify_forecasting", "real_dataset_taskify"),
    },
    "TaskDataset": {
        "group": "containers",
        "summary": "任务专有数据集，显式提供 X/y/masks/aux/schema。",
        "when_to_use": "模型训练、评测、agent 消费或资产保存需要稳定协议时。",
        "related": ("SeriesDataset", "TaskSpec", "build_task_context"),
        "example_ids": ("taskify_forecasting", "causal_response", "policy_value_estimation"),
    },
    "TaskSpec": {
        "group": "containers",
        "summary": "任务语义配置对象，用于描述 taskification 的窗口、步长、horizon 等参数。",
        "when_to_use": "你想把任务设定本身保存下来，给团队或 agent 复用时。",
        "related": ("TaskDataset", "taskify_dataset"),
        "example_ids": ("taskify_forecasting",),
    },
    "describe_series": {
        "group": "analysis",
        "summary": "对单条序列做结构描述，输出 tags、scores 和统计特征。",
        "when_to_use": "拿到真实或外部数据后，想先判断趋势、周期、缺失、随机游走等结构时。",
        "related": ("explain_series", "generate_eda_report", "suggest_spec"),
        "example_ids": ("real_series_eda",),
    },
    "describe_dataset": {
        "group": "analysis",
        "summary": "对整个数据集做聚合描述，输出覆盖率、签名频次和长度统计。",
        "when_to_use": "需要盘点 benchmark 覆盖、真实数据集偏置或结构分桶时。",
        "related": ("generate_dataset_eda_report", "SeriesDataset"),
        "example_ids": ("dataset_eda",),
    },
    "infer_structure_tags": {
        "group": "analysis",
        "summary": "把描述特征转成结构 taxonomy tags。",
        "when_to_use": "你想把真实数据粗归类，或者给 dataset card / agent context 加标签时。",
        "related": ("describe_series", "suggest_spec"),
        "example_ids": ("real_series_eda",),
    },
    "suggest_spec": {
        "group": "analysis",
        "summary": "根据真实序列描述构造一个可执行的 spec seed。",
        "when_to_use": "需要从真实数据反推一个 synthetic matching baseline 时。",
        "related": ("describe_series", "generate_series", "SeriesSpec"),
        "example_ids": ("real_series_eda", "spec_from_real_series"),
    },
    "explain_series": {
        "group": "analysis",
        "summary": "把结构描述翻译成面向人类的解释性 bullet。",
        "when_to_use": "要给报告、agent 或团队沟通输出自然语言解释时。",
        "related": ("describe_series", "generate_eda_report"),
        "example_ids": ("real_series_eda",),
    },
    "generate_eda_report": {
        "group": "analysis",
        "summary": "生成单条序列的 HTML EDA 报告。",
        "when_to_use": "需要漂亮直观的可视化和解释文字来描述一个时间序列时。",
        "related": ("describe_series", "generate_dataset_eda_report"),
        "example_ids": ("real_series_eda", "external_rollout_wrap"),
    },
    "generate_dataset_eda_report": {
        "group": "analysis",
        "summary": "生成整个数据集的 HTML EDA 报告。",
        "when_to_use": "需要面向扩散和沟通的一页式数据集画像时。",
        "related": ("describe_dataset", "SeriesDataset"),
        "example_ids": ("dataset_eda",),
    },
    "generate_linked_eda_bundle": {
        "group": "analysis",
        "summary": "生成可分享的 linked EDA bundle：report.html + docs site + route manifest。",
        "when_to_use": "你希望把 EDA 报告和 landing / quickstart / cookbook / API / FAQ 打成一个可直接分享的目录时。",
        "related": ("generate_eda_report", "generate_docs_site", "build_eda_resource_hub"),
        "example_ids": ("real_series_eda",),
    },
    "generate_linked_dataset_eda_bundle": {
        "group": "analysis",
        "summary": "生成数据集级 linked EDA bundle：report.html + docs site + route manifest。",
        "when_to_use": "你要把整批真实数据的画像、任务建议和文档站一起交付时。",
        "related": ("generate_dataset_eda_report", "generate_docs_site", "build_eda_resource_hub"),
        "example_ids": ("dataset_eda",),
    },
    "EDAResourceHub": {
        "group": "agent",
        "summary": "EDA 报告与 docs site 共享的导航中心对象，汇总 page/example/API/FAQ links。",
        "when_to_use": "你希望把真实数据报告变成一份可操作的导航资产，而不只是图和统计时。",
        "related": ("build_eda_resource_hub", "render_eda_resource_hub_markdown", "generate_eda_report"),
        "example_ids": ("agent_context_pack",),
    },
    "EDARoute": {
        "group": "agent",
        "summary": "一条共享的 EDA finding → task/page/example/API/FAQ 路由规则。",
        "when_to_use": "你要维护报告和 docs 站点之间的统一解释层时。",
        "related": ("common_eda_finding_routes", "build_eda_resource_hub"),
        "example_ids": ("docs_site_generation",),
    },
    "FAQEntry": {
        "group": "agent",
        "summary": "FAQ 的结构化条目对象，带稳定 anchor id。",
        "when_to_use": "你想让 EDA 报告中的 FAQ links 可稳定落到具体答案时。",
        "related": ("FAQ_ENTRIES", "generate_docs_site"),
        "example_ids": ("docs_site_generation",),
    },
    "FAQ_ENTRIES": {
        "group": "agent",
        "summary": "结构化 FAQ 条目集合，供 docs site 和 report routing 共用。",
        "when_to_use": "你要统一维护 FAQ 锚点、内容和 report 跳转目标时。",
        "related": ("FAQEntry", "generate_docs_site", "build_eda_resource_hub"),
        "example_ids": ("docs_site_generation",),
    },
    "common_eda_finding_routes": {
        "group": "agent",
        "summary": "返回内置的共享 EDA route map。",
        "when_to_use": "你想程序化查看“什么样的报告 finding 应该跳到哪里”时。",
        "related": ("EDARoute", "build_eda_resource_hub", "generate_docs_site"),
        "example_ids": ("docs_site_generation",),
    },
    "build_eda_resource_hub": {
        "group": "agent",
        "summary": "根据 SeriesDescription / DatasetDescription 构建报告驱动的资源导航中心。",
        "when_to_use": "你已经拿到描述结果，想进一步产出 page/example/API/FAQ 建议时。",
        "related": ("EDAResourceHub", "generate_eda_report", "generate_dataset_eda_report"),
        "example_ids": ("real_series_eda", "dataset_eda"),
    },
    "render_eda_resource_hub_markdown": {
        "group": "agent",
        "summary": "把 EDAResourceHub 渲染成 Markdown。",
        "when_to_use": "你想把报告驱动的建议写进 README、issue 或 agent memo 时。",
        "related": ("build_eda_resource_hub", "save_eda_resource_hub"),
        "example_ids": ("docs_site_generation",),
    },
    "save_eda_resource_hub": {
        "group": "agent",
        "summary": "把 EDAResourceHub 保存成 JSON / Markdown sidecars。",
        "when_to_use": "你想让导出的 report bundle 自带机器可读和人类可读的路由摘要时。",
        "related": ("build_eda_resource_hub", "render_eda_resource_hub_markdown"),
        "example_ids": ("docs_site_generation",),
    },
    "build_series_context": {
        "group": "agent",
        "summary": "把单条序列压缩成低 token 的结构化 context pack。",
        "when_to_use": "你要把 series 交给 agent，但不想粘贴原始数组或长 README 时。",
        "related": ("build_dataset_context", "build_task_context", "AgentContextPack"),
        "example_ids": ("agent_context_pack",),
    },
    "build_dataset_context": {
        "group": "agent",
        "summary": "为 SeriesDataset 生成 compact context、recommended tasks 与 next actions。",
        "when_to_use": "agent 需要先理解数据集长什么样、应该做什么任务时。",
        "related": ("build_task_context", "build_series_dataset_card", "AgentContextPack"),
        "example_ids": ("agent_context_pack",),
    },
    "build_task_context": {
        "group": "agent",
        "summary": "为 TaskDataset 生成紧凑上下文，突出 schema 和训练所需语义。",
        "when_to_use": "agent 要直接消费 forecasting/causal/control 任务数据集时。",
        "related": ("TaskDataset", "build_task_dataset_card"),
        "example_ids": ("agent_context_pack",),
    },
    "build_agent_context": {
        "group": "agent",
        "summary": "根据对象类型统一构建 compact context。",
        "when_to_use": "你不想自己分流 series / dataset / task dataset 时。",
        "related": ("build_series_context", "build_dataset_context", "build_task_context"),
        "example_ids": ("agent_context_pack",),
    },
    "render_context_markdown": {
        "group": "agent",
        "summary": "把 context pack 渲染成更适合 prompt 或文档的 Markdown。",
        "when_to_use": "你需要把 compact context 保存到 README、PR、Issue 或 prompt 里时。",
        "related": ("AgentContextPack", "build_agent_context"),
        "example_ids": ("agent_context_pack",),
    },
    "AgentContextPack": {
        "group": "agent",
        "summary": "面向 agent 的紧凑上下文容器，包含 compact schema、narrative、next actions 和 example IDs。",
        "when_to_use": "需要在少 token 下稳定传达数据资产语义时。",
        "related": ("build_agent_context", "ArtifactCard"),
        "example_ids": ("agent_context_pack",),
    },
    "build_series_dataset_card": {
        "group": "agent",
        "summary": "为 SeriesDataset 生成 README/JSON 卡片。",
        "when_to_use": "保存基础数据集时，需要让人和 agent 一眼看懂用途与 caveats。",
        "related": ("ArtifactCard", "build_dataset_context"),
        "example_ids": ("dataset_cards",),
    },
    "build_task_dataset_card": {
        "group": "agent",
        "summary": "为 TaskDataset 生成 README/JSON 卡片。",
        "when_to_use": "保存任务数据集并希望 agent 直接读取 schema 与 quickstart 时。",
        "related": ("ArtifactCard", "build_task_context"),
        "example_ids": ("dataset_cards",),
    },
    "ArtifactCard": {
        "group": "agent",
        "summary": "用于保存 intended use、quickstart、caveats 和关键字段的资产说明书。",
        "when_to_use": "要让数据资产自己带说明书，并降低沟通成本时。",
        "related": ("build_series_dataset_card", "build_task_dataset_card"),
        "example_ids": ("dataset_cards",),
    },
    "example_catalog": {
        "group": "agent",
        "summary": "返回内置案例目录。",
        "when_to_use": "你希望把自然语言目标映射到最短可运行代码时。",
        "related": ("recommend_examples", "generate_docs_site"),
        "example_ids": ("docs_site_generation",),
    },
    "recommend_examples": {
        "group": "agent",
        "summary": "根据自然语言意图推荐最相关的案例。",
        "when_to_use": "agent 或新用户不知道该从哪个案例起步时。",
        "related": ("example_catalog", "generate_docs_site"),
        "example_ids": ("docs_site_generation", "agent_context_pack"),
    },
    "build_api_reference": {
        "group": "agent",
        "summary": "生成完整 API 清单、分类、签名和简短说明。",
        "when_to_use": "你要补充 API 文档、让 agent 低 token 地理解库表面，或生成参考页面时。",
        "related": ("APIReference", "render_api_reference_markdown", "save_api_reference"),
        "example_ids": ("api_reference_overview",),
    },
    "render_api_reference_markdown": {
        "group": "agent",
        "summary": "把 APIReference 渲染成 Markdown。",
        "when_to_use": "你需要把 API 参考嵌入 README、文档或 PR 时。",
        "related": ("build_api_reference", "save_api_reference"),
        "example_ids": ("api_reference_overview",),
    },
    "save_api_reference": {
        "group": "agent",
        "summary": "把 APIReference 保存成 JSON 或 Markdown。",
        "when_to_use": "你想把 API manifest 随 release 一起分发时。",
        "related": ("build_api_reference", "render_api_reference_markdown"),
        "example_ids": ("api_reference_overview",),
    },
    "APIReference": {
        "group": "agent",
        "summary": "完整 API 参考对象，包含分类后的公开符号。",
        "when_to_use": "你要程序化访问库的表层接口，或构建静态 API 页面时。",
        "related": ("build_api_reference", "APICategory", "APISymbol"),
        "example_ids": ("api_reference_overview",),
    },
    "generate_docs_site": {
        "group": "agent",
        "summary": "一键生成包含首页、快速上手、案例库、任务化、Agent Playbook、扩散与承载和 API 的静态站点。",
        "when_to_use": "你要发布、扩散、教学，或者把库做成可直接托管的内容站时。",
        "related": ("DocsSiteResult", "example_catalog", "build_api_reference"),
        "example_ids": ("docs_site_generation",),
    },
    "DocsSiteResult": {
        "group": "agent",
        "summary": "静态站点生成结果，记录页面、案例页、API 页和搜索索引。",
        "when_to_use": "你要自动化发布 docs bundle、或检查生成内容完整性时。",
        "related": ("generate_docs_site",),
        "example_ids": ("docs_site_generation",),
    },
    "InterventionSpec": {
        "group": "causal_control",
        "summary": "描述干预目标、区间、索引和值。",
        "when_to_use": "需要对 input/state/output/treatment 注入干预时。",
        "related": ("with_intervention", "generate_counterfactual_pair"),
        "example_ids": ("policy_counterfactual", "intervention_detection"),
    },
    "ConstantPolicy": {
        "group": "causal_control",
        "summary": "恒定动作策略。",
        "when_to_use": "做最简单的 policy baseline 或 counterfactual 对照时。",
        "related": ("PiecewiseConstantPolicy", "LinearFeedbackPolicy"),
        "example_ids": ("policy_counterfactual", "policy_value_estimation"),
    },
    "PiecewiseConstantPolicy": {
        "group": "causal_control",
        "summary": "分段常值策略。",
        "when_to_use": "你要模拟时段化策略或离线 policy schedule 时。",
        "related": ("ConstantPolicy", "ThresholdPolicy"),
        "example_ids": ("policy_value_estimation",),
    },
    "LinearFeedbackPolicy": {
        "group": "causal_control",
        "summary": "线性反馈策略。",
        "when_to_use": "做简单控制、反事实 policy 比较或闭环 state feedback 模拟时。",
        "related": ("PolicyControlledStateSpace", "generate_counterfactual_pair"),
        "example_ids": ("policy_counterfactual",),
    },
    "ThresholdPolicy": {
        "group": "causal_control",
        "summary": "阈值触发策略。",
        "when_to_use": "需要事件触发或开关式控制策略时。",
        "related": ("EventTriggeredController", "InterventionSpec"),
        "example_ids": ("intervention_detection",),
    },
    "with_intervention": {
        "group": "causal_control",
        "summary": "给 spec 注入 intervention 并返回新 spec。",
        "when_to_use": "你想在不改原 spec 的前提下快速生成干预版本时。",
        "related": ("InterventionSpec", "generate_counterfactual_pair"),
        "example_ids": ("policy_counterfactual",),
    },
    "with_policy": {
        "group": "causal_control",
        "summary": "给带控制动态的 spec 挂接新 policy。",
        "when_to_use": "你想比较多个策略下同一系统的 rollout 时。",
        "related": ("LinearFeedbackPolicy", "generate_counterfactual_pair"),
        "example_ids": ("policy_counterfactual",),
    },
    "generate_counterfactual_pair": {
        "group": "causal_control",
        "summary": "生成 factual / counterfactual 成对序列。",
        "when_to_use": "做干预比较、policy 比较、反事实评估时。",
        "related": ("with_intervention", "with_policy", "CounterfactualPair"),
        "example_ids": ("policy_counterfactual",),
    },
    "CounterfactualPair": {
        "group": "causal_control",
        "summary": "打包 factual 与 counterfactual 两条对照序列。",
        "when_to_use": "agent 或人类需要以最少上下文理解“改了什么、结果差多少”时。",
        "related": ("generate_counterfactual_pair",),
        "example_ids": ("policy_counterfactual",),
    },
    "Add": {"group": "operators", "summary": "把多个组件加和。", "when_to_use": "最常见的 trend + seasonal + noise 组合。", "related": ("Multiply", "Convolve", "TimeWarp"), "example_ids": ("quickstart_univariate",)},
    "Multiply": {"group": "operators", "summary": "按点相乘实现调制或门控。", "when_to_use": "需要振幅调制或状态依赖放大时。", "related": ("Add", "TimeWarp"), "example_ids": ("modulated_seasonality",)},
    "Convolve": {"group": "operators", "summary": "通过卷积施加系统响应。", "when_to_use": "需要传感器响应、滤波或脉冲响应时。", "related": ("Add",), "example_ids": ("bursty_response_pipeline",)},
    "TimeWarp": {"group": "operators", "summary": "做时间轴拉伸、压缩或抖动。", "when_to_use": "需要 quasi-periodic、时变速度或对齐扰动时。", "related": ("QuasiPeriodicSeasonality",), "example_ids": ("irregular_sampling_profile",)},
    "Stack": {"group": "operators", "summary": "把多个组件沿通道维堆叠成多变量序列。", "when_to_use": "需要多通道输入、MIMO 参考轨迹或 multivariate series 时。", "related": ("JointServoMIMO", "LinearStateSpace"), "example_ids": ("mimo_joint_servo",)},
    "RegularSampling": {"group": "operators", "summary": "规则采样观测模型。", "when_to_use": "默认规则采样或构造与 irregular 对照实验时。", "related": ("IrregularSampling",), "example_ids": ("robust_observation_variants",)},
    "IrregularSampling": {"group": "operators", "summary": "不规则采样观测模型。", "when_to_use": "模拟真实监测、医疗、事件驱动或异步采样时。", "related": ("RegularSampling", "BlockMissing"), "example_ids": ("robust_observation_variants",)},
    "BlockMissing": {"group": "operators", "summary": "块状缺失机制。", "when_to_use": "需要更真实的连续缺失而不是独立点缺失时。", "related": ("IrregularSampling",), "example_ids": ("robust_observation_variants",)},
    "MeasurementNoise": {"group": "operators", "summary": "测量噪声观测模型。", "when_to_use": "要区分过程噪声与观测噪声时。", "related": ("WhiteGaussianNoise", "ObservationSpec"), "example_ids": ("robust_observation_variants",)},
    "Clamp": {"group": "operators", "summary": "对观测值做裁剪/饱和。", "when_to_use": "模拟传感器饱和、幅值限制时。", "related": ("Downsample",), "example_ids": ("robust_observation_variants",)},
    "Downsample": {"group": "operators", "summary": "降采样观测模型。", "when_to_use": "模拟低频设备、聚合统计或多率观测时。", "related": ("RegularSampling",), "example_ids": ("robust_observation_variants",)},
}


# English-first copy for the public surface. The older overrides remain useful for
# grouping, related APIs, example routing, and the Chinese mirror.
_EN_COPY: dict[str, dict[str, Any]] = {
    "build_demo_dataset": {
        "group": "workflow",
        "summary": "Build one of the built-in showcase datasets, including real public ECG, macro, climate, and sunspot demos plus optional synthetic bundles.",
        "when_to_use": "Use this when you need a credible demo asset for README screenshots, workshops, or public docs.",
        "related": ("demo", "demo_scenario_catalog"),
    },
    "demo_scenario_catalog": {
        "group": "agent",
        "summary": "List the built-in demo scenarios and their narratives, with real public demos surfaced first.",
        "when_to_use": "Use this when docs, starter kits, or CLI help should explain the available flagship demos.",
        "related": ("build_demo_dataset", "demo"),
    },
    "generate_series": {
        "summary": "Compile a spec or a list of components into one GeneratedSeries.",
        "when_to_use": "Use this when you want the shortest path from an idea to one inspectable sequence.",
        "why_exists": "Researchers need a one-line entry point for validating structural ideas before they generate larger datasets.",
        "works_in": ("jupyter-notebook", "python-script"),
        "scenario_ids": ("teaching-and-onboarding", "real-data-profile-and-taskify"),
        "summary_zh": "把 spec 或 component 列表编译成一条 GeneratedSeries。",
        "when_to_use_zh": "当你想从一个结构想法最快得到一条可检查序列时用它。",
        "why_exists_zh": "研究者需要一个一行就能完成的入口，先验证结构想法，再决定是否生成更大的数据集。",
    },
    "compose_series": {
        "summary": "Build one series from a short additive component list without writing a full SeriesSpec.",
        "when_to_use": "Use this when you want a fast, readable composition sketch.",
        "why_exists": "Many users need a minimal composition surface before they are ready for spec-first workflows.",
        "works_in": ("jupyter-notebook", "python-script"),
        "scenario_ids": ("teaching-and-onboarding",),
    },
    "generate_series_dataset": {
        "summary": "Generate a reusable base SeriesDataset instead of locking yourself into one task too early.",
        "when_to_use": "Use this when the dataset should later feed forecasting, classification, control, or causal tasks.",
        "why_exists": "The library is built around reusable data assets first and task views second.",
        "works_in": ("python-script", "headless-server", "jupyter-notebook"),
        "scenario_ids": ("synthetic-benchmark-design", "one-base-dataset-many-tasks"),
        "summary_zh": "生成可复用的基础 SeriesDataset，而不是过早锁死到单一任务。",
        "when_to_use_zh": "当同一份数据以后还要服务 forecasting、classification、control 或 causal 任务时用它。",
        "why_exists_zh": "这个库的核心设计就是先有可复用数据资产，再派生任务视图。",
    },
    "taskify_dataset": {
        "summary": "Turn a SeriesDataset into a task-specific TaskDataset with explicit X, y, masks, aux, and schema.",
        "when_to_use": "Use this when you want one base dataset to support several tasks without losing semantics.",
        "why_exists": "Taskification is the bridge between reusable sequence assets and trainable protocols.",
        "works_in": ("python-script", "jupyter-notebook", "llm-agent-workflow"),
        "scenario_ids": ("one-base-dataset-many-tasks", "synthetic-benchmark-design", "control-rollout-analysis"),
        "summary_zh": "把 SeriesDataset 转成带显式 X、y、masks、aux 和 schema 的 TaskDataset。",
        "when_to_use_zh": "当你希望一份基础数据集支持多个任务，同时不丢失语义时用它。",
        "why_exists_zh": "任务化是可复用序列资产和可训练协议之间的桥。",
    },
    "generate_dataset": {
        "summary": "Generate a task dataset in one step when the task is already known.",
        "when_to_use": "Use this when you already know the task type and want the shortest path to training-ready arrays.",
        "why_exists": "Some workflows value speed over reuse; this API keeps that path simple.",
        "works_in": ("python-script", "headless-server", "jupyter-notebook"),
        "scenario_ids": ("synthetic-benchmark-design", "causal-and-counterfactual-evaluation"),
        "summary_zh": "当任务已经确定时，一步生成任务数据集。",
        "when_to_use_zh": "当你已经知道任务类型，只想最快拿到可训练数组时用它。",
        "why_exists_zh": "有些工作流更重视速度而不是复用，因此这个接口把那条路径保持得很简单。",
    },
    "SeriesSpec": {
        "summary": "The executable specification of a time-series structure program.",
        "when_to_use": "Use this when reproducibility, serialization, and structural clarity matter.",
        "why_exists": "Specs make structural assumptions explicit and portable across teams and experiments.",
        "works_in": ("python-script", "jupyter-notebook", "ci-pipeline"),
        "scenario_ids": ("synthetic-benchmark-design", "causal-and-counterfactual-evaluation"),
    },
    "ObservationSpec": {
        "summary": "The observation layer: sampling, missingness, measurement noise, and post-observation transforms.",
        "when_to_use": "Use this when the measurement process is part of the scientific question or the robustness test.",
        "why_exists": "Real datasets often differ more in how they are observed than in the latent process itself.",
        "works_in": ("python-script", "jupyter-notebook"),
        "scenario_ids": ("synthetic-benchmark-design", "real-data-profile-and-taskify"),
    },
    "GeneratedSeries": {
        "summary": "One generated or wrapped sequence together with time, spec, and trace.",
        "when_to_use": "Use this when you want to inspect one sample deeply before scaling up.",
        "why_exists": "Single-series inspection is the safest way to debug structure, trace, and observation choices.",
        "works_in": ("jupyter-notebook", "python-script"),
        "scenario_ids": ("teaching-and-onboarding", "real-data-profile-and-taskify"),
    },
    "SeriesDataset": {
        "summary": "The reusable base dataset layer for raw time-series assets plus metadata and optional trace.",
        "when_to_use": "Use this when a dataset should survive beyond one task or one notebook.",
        "why_exists": "A stable asset layer lets different tasks share the same underlying data and semantics.",
        "works_in": ("python-script", "headless-server", "llm-agent-workflow"),
        "scenario_ids": ("one-base-dataset-many-tasks", "synthetic-benchmark-design", "agent-grounded-assets"),
    },
    "TaskDataset": {
        "summary": "A task-specific dataset with explicit X, y, masks, aux, and schema.",
        "when_to_use": "Use this when you are ready to train, evaluate, export, or hand the dataset to an agent.",
        "why_exists": "Downstream consumers need stable semantics, not only arrays.",
        "works_in": ("python-script", "headless-server", "llm-agent-workflow", "ci-pipeline"),
        "scenario_ids": ("one-base-dataset-many-tasks", "agent-grounded-assets"),
    },
    "describe_series": {
        "summary": "Describe one time series with interpretable tags, scores, and structure hints.",
        "when_to_use": "Use this before choosing a task or model for real data.",
        "why_exists": "Many projects fail because the task is chosen before the data is understood.",
        "works_in": ("jupyter-notebook", "python-script", "simulator-or-data-lake"),
        "scenario_ids": ("real-data-profile-and-taskify", "control-rollout-analysis"),
        "summary_zh": "对单条时间序列做可解释的标签、分数和结构提示分析。",
        "when_to_use_zh": "在为真实数据选任务或模型之前先用它。",
        "why_exists_zh": "很多项目失败，是因为任务先于数据理解被决定。",
    },
    "describe_dataset": {
        "summary": "Describe a whole dataset through coverage, signatures, and aggregate structure counts.",
        "when_to_use": "Use this when you need to understand a dataset inventory, a benchmark suite, or population-level bias.",
        "why_exists": "Dataset-level decisions need aggregate structure evidence, not only one-series inspection.",
        "works_in": ("python-script", "jupyter-notebook", "headless-server"),
        "scenario_ids": ("synthetic-benchmark-design", "real-data-profile-and-taskify"),
    },
    "suggest_spec": {
        "summary": "Build a first executable spec seed from the description of real data.",
        "when_to_use": "Use this when you want a synthetic match or a starting point for structure-aware simulation.",
        "why_exists": "A useful library should translate from real data back into executable assumptions, not only generate from scratch.",
        "works_in": ("jupyter-notebook", "python-script"),
        "scenario_ids": ("real-data-profile-and-taskify", "synthetic-benchmark-design"),
    },
    "generate_eda_report": {
        "summary": "Render an HTML EDA report that explains one series and routes the user to tasks, examples, APIs, and FAQ entries.",
        "when_to_use": "Use this when the data should explain itself to humans before you open a model or a code search tab.",
        "why_exists": "EDA is not just plotting; it is the first communication layer for a dataset and the bridge into the rest of the package.",
        "works_in": ("jupyter-notebook", "python-script", "static-docs-site", "simulator-or-data-lake"),
        "scenario_ids": ("real-data-profile-and-taskify", "control-rollout-analysis", "docs-launch-and-adoption"),
        "summary_zh": "生成一份 HTML EDA 报告，用来解释单条序列，并把用户路由到任务、案例、API 和 FAQ。",
        "when_to_use_zh": "当你希望数据先向人解释自己，再去打开模型或搜代码时用它。",
        "why_exists_zh": "EDA 不只是画图；它是数据集的第一层沟通界面，也是进入整个包其余部分的桥。",
    },
    "generate_dataset_eda_report": {
        "summary": "Render an HTML EDA report for a whole dataset, including coverage and routing advice.",
        "when_to_use": "Use this when a collection of series needs to be understood, shared, or triaged before modeling.",
        "why_exists": "Teams often need a one-page dataset explanation before they need another experiment.",
        "works_in": ("python-script", "headless-server", "static-docs-site"),
        "scenario_ids": ("real-data-profile-and-taskify", "synthetic-benchmark-design", "docs-launch-and-adoption"),
    },
    "build_agent_context": {
        "summary": "Build a compact, low-token context pack for a series, dataset, or task.",
        "when_to_use": "Use this when an agent should read the semantics first instead of the raw arrays.",
        "why_exists": "Token budgets are now a practical design constraint for research tooling.",
        "works_in": ("llm-agent-workflow", "python-script", "ci-pipeline"),
        "scenario_ids": ("agent-grounded-assets", "docs-launch-and-adoption"),
    },
    "build_dataset_context": {
        "summary": "Build a compact context for a base SeriesDataset.",
        "when_to_use": "Use this when a reusable asset has to travel through an agent or a narrow prompt budget.",
        "why_exists": "Base datasets need their own summary layer, not only task-specific summaries.",
        "works_in": ("llm-agent-workflow", "python-script"),
        "scenario_ids": ("agent-grounded-assets", "one-base-dataset-many-tasks"),
    },
    "build_task_context": {
        "summary": "Build a compact context for a task-specific dataset with schema-aware fields.",
        "when_to_use": "Use this when a model-training asset must be consumed by another system or assistant.",
        "why_exists": "Task datasets need an explicit bridge into downstream automation.",
        "works_in": ("llm-agent-workflow", "python-script", "ci-pipeline"),
        "scenario_ids": ("agent-grounded-assets",),
    },
    "HandoffIndex": {
        "summary": "A tiny, agent-first entrypoint that says what to open first and what the next action is.",
        "when_to_use": "Use this when an agent or teammate should not start from the full bundle JSON.",
        "why_exists": "The first artifact in a handoff should be small, path-oriented, and low-token.",
        "works_in": ("llm-agent-workflow", "python-script", "ci-pipeline"),
        "scenario_ids": ("agent-grounded-assets", "docs-launch-and-adoption"),
    },
    "DatasetHandoffBundle": {
        "summary": "A unified report + card + context + manifest package for moving a dataset asset forward.",
        "when_to_use": "Use this when a time-series dataset has to be explained, shared, and handed to the next person or agent in one predictable directory.",
        "why_exists": "The shortest public happy path should lead to a concrete artifact bundle, not only to scattered helper functions.",
        "works_in": ("python-script", "llm-agent-workflow", "static-docs-site"),
        "scenario_ids": ("agent-grounded-assets", "docs-launch-and-adoption", "real-data-profile-and-taskify"),
    },
    "build_dataset_handoff_bundle": {
        "summary": "Build the shortest outcome-first bundle for a dataset: report, context, card, handoff index, manifest, and explicit next actions.",
        "when_to_use": "Use this when you already have a dataset asset and want one call that produces the report-first handoff surface.",
        "why_exists": "Users should not have to stitch together EDA, cards, contexts, and manifests by hand for the common case.",
        "works_in": ("python-script", "headless-server", "llm-agent-workflow"),
        "scenario_ids": ("agent-grounded-assets", "real-data-profile-and-taskify", "docs-launch-and-adoption"),
        "summary_zh": "为数据集生成最短的 outcome-first bundle：report、context、card、manifest 和明确 next actions。",
        "when_to_use_zh": "当你已经有一份 dataset asset，并且希望通过一次调用拿到 report-first 的 handoff 表层时用它。",
        "why_exists_zh": "面对最常见场景时，用户不该自己手动拼接 EDA、card、context 和 manifest。",
    },
    "render_handoff_index_markdown": {
        "summary": "Render the tiny handoff index into a Markdown quick-open guide.",
        "when_to_use": "Use this when README text should point humans or agents to the smallest correct first artifact.",
        "why_exists": "The compact entrypoint deserves its own human-readable form, not only JSON.",
        "works_in": ("python-script", "static-docs-site", "llm-agent-workflow"),
        "scenario_ids": ("agent-grounded-assets", "docs-launch-and-adoption"),
    },
    "render_dataset_handoff_markdown": {
        "summary": "Render a handoff bundle into a README-style Markdown view.",
        "when_to_use": "Use this when the bundle should be human-readable in GitHub, docs, or file previews.",
        "why_exists": "The same bundle should serve both human readers and machine consumers.",
        "works_in": ("python-script", "static-docs-site", "llm-agent-workflow"),
        "scenario_ids": ("docs-launch-and-adoption", "agent-grounded-assets"),
    },
    "save_dataset_handoff_bundle": {
        "summary": "Save a handoff bundle to disk as JSON, Markdown, and README assets.",
        "when_to_use": "Use this when you have already built a bundle in memory and want to persist it into a shareable directory.",
        "why_exists": "A first-class asset layer needs a first-class persistence path.",
        "works_in": ("python-script", "headless-server", "ci-pipeline"),
        "scenario_ids": ("agent-grounded-assets", "docs-launch-and-adoption"),
    },
    "build_api_reference": {
        "summary": "Build a structured inventory of the public API surface, grouped by workflow and use case.",
        "when_to_use": "Use this when you want humans and agents to navigate the library by intent instead of by source tree.",
        "why_exists": "Public APIs need rationale, grouping, and machine-readable structure to scale support and adoption.",
        "works_in": ("ci-pipeline", "static-docs-site", "llm-agent-workflow"),
        "scenario_ids": ("docs-launch-and-adoption", "agent-grounded-assets"),
    },
    "generate_docs_site": {
        "summary": "Generate a static bilingual docs site with landing pages, scenarios, examples, API, and FAQ.",
        "when_to_use": "Use this when the package has to be understandable outside the source repository.",
        "why_exists": "Adoption depends on an explorable public surface, not only on code quality.",
        "works_in": ("static-docs-site", "ci-pipeline", "python-script"),
        "scenario_ids": ("docs-launch-and-adoption", "teaching-and-onboarding"),
    },
    "playbook_catalog": {
        "summary": "Return goal-first workflow playbooks that explain when TSDataForge is the right tool and what to run first.",
        "when_to_use": "Use this when you want a user-facing or agent-facing map from intent to workflow.",
        "why_exists": "Most users think in goals, not modules; playbooks keep the public surface aligned with that reality.",
        "works_in": ("static-docs-site", "llm-agent-workflow", "python-script"),
        "scenario_ids": ("teaching-and-onboarding", "docs-launch-and-adoption"),
    },
    "starter_catalog": {
        "summary": "Return starter project templates that package scripts, notebooks, and manifests around common entry paths.",
        "when_to_use": "Use this when users need a runnable project layout instead of another page of prose.",
        "why_exists": "Starter kits shorten time-to-success and make adoption more repeatable across teams.",
        "works_in": ("static-docs-site", "python-script", "llm-agent-workflow"),
        "scenario_ids": ("teaching-and-onboarding", "docs-launch-and-adoption"),
    },
    "recommend_playbooks": {
        "summary": "Recommend a small set of playbooks for a natural-language goal.",
        "when_to_use": "Use this in onboarding flows, agents, or command selection UIs.",
        "why_exists": "A short ranked list of workflows is often more useful than a long explanation or a flat menu.",
        "works_in": ("llm-agent-workflow", "static-docs-site", "python-script"),
        "scenario_ids": ("teaching-and-onboarding", "agent-grounded-assets"),
    },
    "recommend_starters": {
        "summary": "Recommend starter kits for a natural-language use case or environment.",
        "when_to_use": "Use this when the next best action is to hand someone a runnable template.",
        "why_exists": "The distance between reading docs and running code is a major adoption bottleneck.",
        "works_in": ("llm-agent-workflow", "static-docs-site", "python-script"),
        "scenario_ids": ("teaching-and-onboarding", "docs-launch-and-adoption"),
    },
    "export_tutorial_notebooks": {
        "summary": "Export tutorial tracks as notebook and script assets derived from tested examples.",
        "when_to_use": "Use this when tutorials should be runnable locally or distributed as downloadable assets.",
        "why_exists": "Many users learn faster from a notebook than from an API page, but those notebooks should still stay tied to tested examples.",
        "works_in": ("python-script", "ci-pipeline", "static-docs-site"),
        "scenario_ids": ("teaching-and-onboarding", "docs-launch-and-adoption"),
    },
    "create_starter_project": {
        "summary": "Render one starter kit into a project directory with README, scripts, notebooks, and a manifest.",
        "when_to_use": "Use this when you want to hand someone a concrete project scaffold for a known workflow.",
        "why_exists": "A shareable scaffold often converts curiosity into actual usage faster than documentation alone.",
        "works_in": ("python-script", "static-docs-site", "llm-agent-workflow"),
        "scenario_ids": ("teaching-and-onboarding", "docs-launch-and-adoption", "agent-grounded-assets"),
    },
    "wrap_external_series": {
        "summary": "Wrap external rollouts or real-world arrays so TSDataForge can add structure-aware reports and taskification on top.",
        "when_to_use": "Use this when the time series come from another simulator or production system.",
        "why_exists": "The package should integrate with external data sources instead of rebuilding every upstream stack.",
        "works_in": ("simulator-or-data-lake", "python-script", "jupyter-notebook"),
        "scenario_ids": ("control-rollout-analysis", "real-data-profile-and-taskify"),
    },
    "generate_counterfactual_pair": {
        "summary": "Generate a factual and counterfactual pair under a changed policy or intervention.",
        "when_to_use": "Use this when you need a concrete before/after comparison rather than an abstract causal story.",
        "why_exists": "Counterfactual work becomes much easier to explain when it yields paired artifacts.",
        "works_in": ("python-script", "jupyter-notebook", "headless-server"),
        "scenario_ids": ("causal-and-counterfactual-evaluation",),
    },
}


_GENERIC_SUMMARIES: dict[str, str] = {
    "workflow": "A workflow-level public entry point.",
    "containers": "A core container or specification object that keeps the library composable.",
    "analysis": "An analysis or explanation surface for understanding data before modelling.",
    "agent": "A low-token or publication-facing surface for agents, docs, or artifacts.",
    "causal_control": "A control, intervention, policy, or counterfactual interface.",
    "operators": "A composition or observation component for building structured series.",
    "primitives": "A structural primitive or dynamic generator used inside specs.",
}

_GENERIC_WHY: dict[str, str] = {
    "workflow": "The library needs short public entry points that match how researchers actually work.",
    "containers": "Stable containers keep data, semantics, and metadata together across tasks and tools.",
    "analysis": "Understanding structure before modelling prevents many downstream mistakes.",
    "agent": "Public assets need to be compact, explainable, and machine-readable.",
    "causal_control": "Control and causal workflows need explicit intervention and policy semantics.",
    "operators": "Latent structure and observation conditions should be composable instead of hard-coded together.",
    "primitives": "Researchers need reusable building blocks instead of one-off generators.",
}

_GENERIC_WHY_ZH: dict[str, str] = {
    "workflow": "这个库需要和真实研究工作流匹配的短公共入口。",
    "containers": "稳定的数据容器能让数据、语义和元信息在不同任务和工具之间保持绑定。",
    "analysis": "在建模前理解结构，可以避免许多下游错误。",
    "agent": "公开资产需要足够紧凑、可解释、可被机器读取。",
    "causal_control": "控制和因果工作流需要显式的 intervention 与 policy 语义。",
    "operators": "latent structure 和 observation condition 应该是可组合的，而不是硬编码到一起。",
    "primitives": "研究者需要可复用的结构原语，而不是一次性的生成函数。",
}

_GROUP_ENVIRONMENTS: dict[str, tuple[str, ...]] = {
    "workflow": ("python-script", "jupyter-notebook"),
    "containers": ("python-script", "jupyter-notebook", "llm-agent-workflow"),
    "analysis": ("jupyter-notebook", "python-script", "static-docs-site"),
    "agent": ("llm-agent-workflow", "ci-pipeline", "static-docs-site"),
    "causal_control": ("python-script", "jupyter-notebook", "headless-server"),
    "operators": ("python-script", "jupyter-notebook"),
    "primitives": ("python-script", "jupyter-notebook"),
}

_GROUP_SCENARIOS: dict[str, tuple[str, ...]] = {
    "workflow": ("synthetic-benchmark-design", "one-base-dataset-many-tasks"),
    "containers": ("one-base-dataset-many-tasks", "agent-grounded-assets"),
    "analysis": ("real-data-profile-and-taskify",),
    "agent": ("agent-grounded-assets", "docs-launch-and-adoption"),
    "causal_control": ("control-rollout-analysis", "causal-and-counterfactual-evaluation"),
    "operators": ("synthetic-benchmark-design",),
    "primitives": ("synthetic-benchmark-design",),
}


def _contains_cjk(text: str | None) -> bool:
    return bool(text and _CJK_RE.search(text))


def _infer_group(name: str, module: str) -> str:
    if name in _OVERRIDES:
        return str(_OVERRIDES[name].get("group", "containers"))
    if ".agent" in module:
        return "agent"
    if ".analysis" in module or ".report" in module:
        return "analysis"
    if ".datasets" in module:
        return "containers"
    if ".primitives" in module or ".dynamics" in module:
        return "primitives"
    if ".observation" in module or ".operators" in module:
        return "operators"
    if ".interventions" in module or ".policies" in module or "counterfactual" in module:
        return "causal_control"
    if module.endswith("api") or module.endswith("compiler"):
        return "workflow"
    return "containers"


_DEF_USE = {
    "function": "Call this when the workflow reaches this step.",
    "class": "Instantiate this when you need to keep configuration or semantics explicit.",
}


def _kind(obj: Any) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj):
        return "function"
    return type(obj).__name__.lower()


def _signature(name: str, obj: Any) -> str:
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return name
    return f"{name}{sig}"


def _to_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _summary(name: str, obj: Any, group: str) -> tuple[str, str]:
    if name in _EN_COPY and _EN_COPY[name].get("summary"):
        en = str(_EN_COPY[name]["summary"])
        zh = str(_EN_COPY[name].get("summary_zh", ""))
        return en, zh
    override = _OVERRIDES.get(name, {})
    override_summary = override.get("summary")
    if isinstance(override_summary, str) and override_summary.strip() and not _contains_cjk(override_summary):
        return override_summary.strip(), ""
    doc = inspect.getdoc(obj) or ""
    if doc:
        first = doc.strip().splitlines()[0].strip()
        if first:
            return first, str(override_summary).strip() if isinstance(override_summary, str) and _contains_cjk(override_summary) else ""
    generic = _GENERIC_SUMMARIES.get(group, "Public API exported by tsdataforge.")
    zh = str(override_summary).strip() if isinstance(override_summary, str) and _contains_cjk(override_summary) else ""
    return generic, zh


def _when_to_use(name: str, kind: str) -> tuple[str, str]:
    if name in _EN_COPY and _EN_COPY[name].get("when_to_use"):
        return str(_EN_COPY[name]["when_to_use"]), str(_EN_COPY[name].get("when_to_use_zh", ""))
    override = _OVERRIDES.get(name, {})
    override_text = override.get("when_to_use")
    if isinstance(override_text, str) and override_text.strip() and not _contains_cjk(override_text):
        return override_text.strip(), ""
    zh = str(override_text).strip() if isinstance(override_text, str) and _contains_cjk(override_text) else ""
    return _DEF_USE.get(kind, "Use this when the surrounding workflow points here."), zh


def _why_exists(name: str, group: str) -> tuple[str, str]:
    if name in _EN_COPY and _EN_COPY[name].get("why_exists"):
        return str(_EN_COPY[name]["why_exists"]), str(_EN_COPY[name].get("why_exists_zh", ""))
    return _GENERIC_WHY.get(group, "This symbol exists to keep the public surface usable."), _GENERIC_WHY_ZH.get(group, "这个符号存在，是为了让公共表层更好用。")


def _works_in(name: str, group: str) -> tuple[str, ...]:
    if name in _EN_COPY and _EN_COPY[name].get("works_in"):
        return _to_tuple(_EN_COPY[name].get("works_in"))
    return _GROUP_ENVIRONMENTS.get(group, ("python-script",))


def _scenario_ids(name: str, group: str) -> tuple[str, ...]:
    if name in _EN_COPY and _EN_COPY[name].get("scenario_ids"):
        return _to_tuple(_EN_COPY[name].get("scenario_ids"))
    return _GROUP_SCENARIOS.get(group, ())


def build_api_reference(mode: str = "public") -> APIReference:
    """Build a categorized reference for the public ``tsdataforge`` API.

    The reference is designed for both human docs and low-token agent
    consumption: every symbol gets a stable category, a signature, a concise
    explanation, a rationale, suggested environments, and related examples.
    """

    import tsdataforge as tdf

    categories: dict[str, list[APISymbol]] = {key: [] for key in _GROUPS}
    if str(mode).lower() == "full":
        exported = list(getattr(tdf, "_FULL_EXPORTS", getattr(tdf, "__all__", [])))
    else:
        exported = list(getattr(tdf, "__all__", []))

    for name in exported:
        try:
            obj = getattr(tdf, name)
        except AttributeError:
            continue
        module = getattr(obj, "__module__", "tsdataforge")
        override = _OVERRIDES.get(name, {})
        group = str(override.get("group", _infer_group(name, module)))
        kind = _kind(obj)
        summary, summary_zh = _summary(name, obj, group)
        when_to_use, when_to_use_zh = _when_to_use(name, kind)
        why_exists, why_exists_zh = _why_exists(name, group)
        symbol = APISymbol(
            name=name,
            kind=kind,
            module=module,
            signature=_signature(name, obj),
            summary=summary,
            when_to_use=when_to_use,
            why_exists=why_exists,
            works_in=_works_in(name, group),
            scenario_ids=_scenario_ids(name, group),
            summary_zh=summary_zh,
            when_to_use_zh=when_to_use_zh,
            why_exists_zh=why_exists_zh,
            related=_to_tuple(override.get("related")),
            example_ids=_to_tuple(override.get("example_ids")),
            group=group,
        )
        categories.setdefault(group, []).append(symbol)

    ordered_categories: list[APICategory] = []
    total = 0
    for key, meta in _GROUPS.items():
        items = tuple(sorted(categories.get(key, []), key=lambda item: item.name.lower()))
        total += len(items)
        ordered_categories.append(APICategory(category_id=key, title=meta["title"], summary=meta["summary"], symbols=items))
    return APIReference(version=str(getattr(tdf, "__version__", "unknown")), categories=tuple(ordered_categories), n_symbols=total)


def render_api_reference_markdown(reference: APIReference | None = None, *, mode: str = "public") -> str:
    """Render an API reference into Markdown."""
    ref = reference or build_api_reference(mode=mode)
    return ref.to_markdown()


def save_api_reference(reference: APIReference | None, path: str | Path, *, mode: str = "public") -> None:
    """Save an API reference as JSON or Markdown."""
    ref = reference or build_api_reference(mode=mode)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(ref.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if path.suffix.lower() in {".md", ".markdown"}:
        path.write_text(ref.to_markdown(), encoding="utf-8")
        return
    raise ValueError("Unsupported API reference extension. Use .json or .md")


__all__ = [
    "APISymbol",
    "APICategory",
    "APIReference",
    "build_api_reference",
    "render_api_reference_markdown",
    "save_api_reference",
]



_OVERRIDES.update({
    "compare_series": {"group": "analysis", "summary": "比较两条时间序列的形状、节奏与频域特征。", "when_to_use": "当你想回答两条序列是否‘看起来像同一种动态’时。", "related": ("pairwise_similarity", "find_top_matches", "generate_eda_report"), "example_ids": ("openclaw_stars_similarity", "btc_gold_oil_similarity")},
    "pairwise_similarity": {"group": "analysis", "summary": "为一组命名序列构建成对相似性矩阵。", "when_to_use": "当你想一次性比较一个 panel 里的多条序列时。", "related": ("compare_series", "find_top_matches"), "example_ids": ("github_stars_pairwise_panel", "btc_gold_oil_similarity")},
    "find_top_matches": {"group": "analysis", "summary": "以一条参考序列为中心，给候选集合做相似性排序。", "when_to_use": "当你的故事是‘谁最像这条参考序列’而不是全矩阵比较时。", "related": ("compare_series", "pairwise_similarity"), "example_ids": ("github_stars_pairwise_panel", "btc_gold_oil_similarity")},
    "explain_similarity": {"group": "analysis", "summary": "把相似性结果转成人类可读解释。", "when_to_use": "当你要发布案例、写报告或给 agent 提供简短解释时。", "related": ("compare_series", "SimilarityResult"), "example_ids": ("openclaw_stars_similarity",)},
    "SimilarityResult": {"group": "analysis", "summary": "单对序列比较后的可解释结果对象。", "when_to_use": "当你需要保存 aggregate score、分项指标和解释文本时。", "related": ("compare_series", "SimilarityMatrix")},
    "SimilarityMatrix": {"group": "analysis", "summary": "多条序列的成对相似性矩阵。", "when_to_use": "当你需要 panel 级比较或想做聚类/排序前的快速总览时。", "related": ("pairwise_similarity",)},
    "fetch_github_stars_series": {"group": "analysis", "summary": "抓取 GitHub 仓库 star 历史并转成可分析的时间序列。", "when_to_use": "当你想分析公共仓库的关注度变化、star 动量或热点扩散形状时。", "related": ("pairwise_similarity", "generate_eda_report", "wrap_external_series"), "example_ids": ("openclaw_stars_similarity", "github_stars_pairwise_panel")},
    "fetch_fred_series": {"group": "analysis", "summary": "从 FRED 抓取宏观或商品时间序列。", "when_to_use": "当你要分析黄金、原油或其他 FRED 公共序列时。", "related": ("fetch_coingecko_market_chart", "pairwise_similarity"), "example_ids": ("btc_gold_oil_similarity",)},
    "fetch_coingecko_market_chart": {"group": "analysis", "summary": "从 CoinGecko 抓取加密资产市场图表序列。", "when_to_use": "当你要分析 Bitcoin 等加密资产的价格、成交量或与其他市场序列的关系时。", "related": ("fetch_fred_series", "pairwise_similarity"), "example_ids": ("btc_gold_oil_similarity",)},
})

_EN_COPY.update({
    "compare_series": {
        "summary": "Compare two time series with explainable shape, alignment, and spectral metrics.",
        "when_to_use": "Use this when you want to answer whether two series behave similarly after alignment and normalization.",
        "why_exists": "Attention curves, market moves, and control trajectories often need a transparent baseline similarity method before heavier modeling.",
        "works_in": ("jupyter-notebook", "python-script", "llm-agent-workflow"),
        "scenario_ids": ("live-market-and-attention-analysis", "real-data-profile-and-taskify"),
    },
    "pairwise_similarity": {
        "summary": "Build a pairwise similarity matrix for a named panel of series.",
        "when_to_use": "Use this when you want a fast panel-wide answer to which public signals look most alike.",
        "why_exists": "Many case studies compare several series at once; a matrix is easier to publish and inspect than many isolated pair calls.",
        "works_in": ("jupyter-notebook", "python-script", "static-docs-site"),
        "scenario_ids": ("live-market-and-attention-analysis",),
    },
    "find_top_matches": {
        "summary": "Rank candidate series by similarity to one reference series.",
        "when_to_use": "Use this when the story starts from one anchor signal and asks what else looks most similar.",
        "why_exists": "Reference-first matching is often the clearest storytelling format for public case studies and retrieval workflows.",
        "works_in": ("jupyter-notebook", "python-script", "llm-agent-workflow"),
        "scenario_ids": ("live-market-and-attention-analysis", "agent-grounded-assets"),
    },
    "explain_similarity": {
        "summary": "Return a concise human-readable explanation for a similarity result.",
        "when_to_use": "Use this when you want a short report sentence instead of manually interpreting each metric.",
        "why_exists": "Similarity scores are more useful when they travel with a narrative explanation that humans and agents can both reuse.",
        "works_in": ("python-script", "llm-agent-workflow", "static-docs-site"),
        "scenario_ids": ("live-market-and-attention-analysis", "docs-launch-and-adoption"),
    },
    "SimilarityResult": {
        "summary": "Container for one explainable similarity comparison between two series.",
        "when_to_use": "Use this when you need the score, component metrics, tag overlap, and narrative summary together.",
        "why_exists": "A publishable comparison needs more than one float; it needs evidence that can be saved, rendered, and reused.",
        "works_in": ("python-script", "static-docs-site", "llm-agent-workflow"),
        "scenario_ids": ("live-market-and-attention-analysis",),
    },
    "SimilarityMatrix": {
        "summary": "Container for a pairwise similarity matrix across many named series.",
        "when_to_use": "Use this when a panel-level overview matters more than one isolated pair.",
        "why_exists": "Many adoption-friendly case studies start with a simple matrix that makes relationships immediately visible.",
        "works_in": ("jupyter-notebook", "python-script", "static-docs-site"),
        "scenario_ids": ("live-market-and-attention-analysis",),
    },
    "fetch_github_stars_series": {
        "summary": "Fetch GitHub stargazer history and turn it into a GeneratedSeries with daily and cumulative star traces.",
        "when_to_use": "Use this when you want to analyze repository attention, momentum, or public developer-interest curves.",
        "why_exists": "Public GitHub projects are one of the easiest ways to create high-interest real time-series case studies that many researchers instantly understand.",
        "works_in": ("python-script", "jupyter-notebook", "static-docs-site"),
        "scenario_ids": ("live-market-and-attention-analysis", "docs-launch-and-adoption"),
    },
    "fetch_fred_series": {
        "summary": "Fetch a macro or commodity time series from FRED and wrap it as a GeneratedSeries.",
        "when_to_use": "Use this when you need public gold, oil, rates, or other macro reference series inside the same workflow as synthetic or external data.",
        "why_exists": "Market and macro comparisons are a practical way to demonstrate taskification, EDA, and similarity analysis on data that people already care about.",
        "works_in": ("python-script", "jupyter-notebook"),
        "scenario_ids": ("live-market-and-attention-analysis",),
    },
    "fetch_coingecko_market_chart": {
        "summary": "Fetch historical crypto market-chart data from CoinGecko and wrap it as a GeneratedSeries.",
        "when_to_use": "Use this when you want Bitcoin or other crypto assets to enter the same EDA and similarity workflow as commodities or attention data.",
        "why_exists": "Crypto gives the library a high-interest, globally understood, and continuously updated public signal family for tutorials and demos.",
        "works_in": ("python-script", "jupyter-notebook"),
        "scenario_ids": ("live-market-and-attention-analysis",),
    },
})


_OVERRIDES.update({
    "PackageProfile": {"group": "agent", "summary": "生态定位页中的单个库画像对象。", "when_to_use": "当你想把 package positioning 做成可程序消费的数据，而不是散落在 README 文案里时。", "related": ("PositioningMatrix", "competitor_catalog")},
    "PositioningMatrix": {"group": "agent", "summary": "TSDataForge 与周边库的定位矩阵对象。", "when_to_use": "当你要生成 README、docs 页面、FAQ 或 GitHub 说明中的差异化表格时。", "related": ("build_positioning_matrix", "render_positioning_markdown")},
    "competitor_catalog": {"group": "agent", "summary": "返回围绕 TSDataForge 的生态库画像列表。", "when_to_use": "当你想说明 TSDataForge 与 tsfresh、sktime、Darts、tslearn 等库的关系时。", "related": ("build_positioning_matrix", "recommend_companions")},
    "build_positioning_matrix": {"group": "agent", "summary": "构建生态定位矩阵，供 README、landing 和 docs 复用。", "when_to_use": "当你要把‘这个包和别人有什么不同’说清楚，并且希望网页与说明保持一致时。", "related": ("PositioningMatrix", "render_positioning_markdown", "save_positioning_matrix")},
    "render_positioning_markdown": {"group": "agent", "summary": "把生态定位矩阵渲染成 Markdown。", "when_to_use": "当你要生成 GitHub README 风格的定位说明时。", "related": ("build_positioning_matrix", "save_positioning_matrix")},
    "save_positioning_matrix": {"group": "agent", "summary": "把生态定位矩阵保存成 JSON 或 Markdown。", "when_to_use": "当你要把定位说明变成可分享、可自动化消费的资产时。", "related": ("build_positioning_matrix", "render_positioning_markdown")},
    "recommend_companions": {"group": "agent", "summary": "根据工作流推荐最适合与 TSDataForge 搭配的外部库。", "when_to_use": "当你不想给出‘替代一切’的口号，而是想告诉用户接下来该接哪个库时。", "related": ("competitor_catalog", "build_positioning_matrix")},
})

_EN_COPY.update({
    "PackageProfile": {
        "summary": "A structured profile for one library in the surrounding ecosystem.",
        "when_to_use": "Use this when the positioning story itself should be reusable data rather than scattered prose.",
        "why_exists": "Good GitHub surfaces and docs pages need a stable, auditable source of truth for ecosystem fit.",
        "works_in": ("static-docs-site", "llm-agent-workflow", "python-script"),
        "scenario_ids": ("docs-launch-and-adoption", "agent-grounded-assets"),
    },
    "PositioningMatrix": {
        "summary": "A reusable ecosystem-fit matrix for README, docs, landing pages, and FAQ surfaces.",
        "when_to_use": "Use this when you want the answer to 'why this package, and why not only the others?' to stay consistent across surfaces.",
        "why_exists": "Positioning drifts quickly when it lives only in handwritten copy. A matrix keeps the story stable.",
        "works_in": ("static-docs-site", "llm-agent-workflow", "ci-pipeline"),
        "scenario_ids": ("docs-launch-and-adoption", "agent-grounded-assets"),
    },
    "competitor_catalog": {
        "summary": "Return the curated companion-library map around TSDataForge.",
        "when_to_use": "Use this when you need to explain ecosystem fit instead of making vague all-in-one claims.",
        "why_exists": "Researchers adopt tools faster when they know exactly what the package replaces, what it complements, and what it leaves to others.",
        "works_in": ("static-docs-site", "llm-agent-workflow", "python-script"),
        "scenario_ids": ("docs-launch-and-adoption",),
    },
    "build_positioning_matrix": {
        "summary": "Build the ecosystem-fit matrix that explains how TSDataForge differs from adjacent libraries.",
        "when_to_use": "Use this when README, landing, docs, and FAQ all need the same differentiation story.",
        "why_exists": "Package positioning should be generated from one source of truth, not rewritten differently on every page.",
        "works_in": ("static-docs-site", "ci-pipeline", "llm-agent-workflow"),
        "scenario_ids": ("docs-launch-and-adoption", "agent-grounded-assets"),
    },
    "render_positioning_markdown": {
        "summary": "Render the positioning matrix into GitHub-friendly Markdown.",
        "when_to_use": "Use this when you want ecosystem fit to appear clearly in a README or release asset.",
        "why_exists": "Good README surfaces explain both capability and boundary.",
        "works_in": ("static-docs-site", "ci-pipeline", "python-script"),
        "scenario_ids": ("docs-launch-and-adoption",),
    },
    "save_positioning_matrix": {
        "summary": "Save the positioning matrix as JSON or Markdown for publishing or automation.",
        "when_to_use": "Use this when your public surface or agent workflow should ingest the positioning story as a file.",
        "why_exists": "Positioning becomes more durable when it is saved as a versioned artifact.",
        "works_in": ("ci-pipeline", "python-script", "static-docs-site"),
        "scenario_ids": ("docs-launch-and-adoption",),
    },
    "recommend_companions": {
        "summary": "Recommend which external libraries pair best with TSDataForge for a given workflow.",
        "when_to_use": "Use this when users ask 'what should I use with this for forecasting, feature extraction, or motif mining?'.",
        "why_exists": "The most credible libraries know where to hand users off next.",
        "works_in": ("llm-agent-workflow", "static-docs-site", "python-script"),
        "scenario_ids": ("docs-launch-and-adoption", "agent-grounded-assets"),
    },
})
