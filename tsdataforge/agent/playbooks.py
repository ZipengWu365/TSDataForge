from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Iterable

from .examples import ExampleRecipe, example_catalog
from .tutorials import TutorialTrack, tutorial_catalog


@dataclass(frozen=True)
class Playbook:
    playbook_id: str
    title: str
    summary: str
    promise: str
    when_to_use: str
    steps: tuple[str, ...] = field(default_factory=tuple)
    scenario_ids: tuple[str, ...] = field(default_factory=tuple)
    environment_ids: tuple[str, ...] = field(default_factory=tuple)
    example_ids: tuple[str, ...] = field(default_factory=tuple)
    tutorial_ids: tuple[str, ...] = field(default_factory=tuple)
    api_names: tuple[str, ...] = field(default_factory=tuple)
    starter_id: str = ""
    title_zh: str = ""
    summary_zh: str = ""
    promise_zh: str = ""
    when_to_use_zh: str = ""
    steps_zh: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self, *, language: str = "en") -> dict:
        data = asdict(self)
        lang = (language or "en").lower()
        if lang.startswith("zh"):
            data["title"] = self.title_zh or self.title
            data["summary"] = self.summary_zh or self.summary
            data["promise"] = self.promise_zh or self.promise
            data["when_to_use"] = self.when_to_use_zh or self.when_to_use
            data["steps"] = list(self.steps_zh or self.steps)
        else:
            data["steps"] = list(self.steps)
        return data


@dataclass(frozen=True)
class StarterKit:
    starter_id: str
    title: str
    summary: str
    best_for: tuple[str, ...] = field(default_factory=tuple)
    environment_ids: tuple[str, ...] = field(default_factory=tuple)
    tutorial_ids: tuple[str, ...] = field(default_factory=tuple)
    example_ids: tuple[str, ...] = field(default_factory=tuple)
    api_names: tuple[str, ...] = field(default_factory=tuple)
    generated_files: tuple[str, ...] = field(default_factory=tuple)
    title_zh: str = ""
    summary_zh: str = ""
    best_for_zh: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self, *, language: str = "en") -> dict:
        data = asdict(self)
        lang = (language or "en").lower()
        if lang.startswith("zh"):
            data["title"] = self.title_zh or self.title
            data["summary"] = self.summary_zh or self.summary
            data["best_for"] = list(self.best_for_zh or self.best_for)
        else:
            data["best_for"] = list(self.best_for)
        return data


@dataclass(frozen=True)
class NotebookAsset:
    notebook_id: str
    title: str
    summary: str
    tutorial_id: str
    path_ipynb: str
    path_py: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StarterProjectResult:
    starter_id: str
    output_dir: str
    manifest_path: str
    readme_path: str
    script_paths: tuple[str, ...] = field(default_factory=tuple)
    notebook_paths: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


_PLAYBOOKS: tuple[Playbook, ...] = (
    Playbook(
        playbook_id="first-success",
        title="Get to a meaningful first success in under 10 minutes",
        summary="The shortest guided path for a new user: generate one series, explain it, create one base dataset, taskify it, and save the result.",
        promise="You will leave with one dataset asset and one mental model instead of a pile of disconnected snippets.",
        when_to_use="Use this when you are new to the package or teaching it to someone else.",
        steps=(
            "Generate one sequence with trend + seasonality + noise.",
            "Inspect trace or generate a short EDA report to see what the library preserves.",
            "Create a base SeriesDataset and turn it into a forecasting task.",
            "Save the TaskDataset so README, schema, and cards travel with the data.",
        ),
        scenario_ids=("teaching-and-onboarding", "one-base-dataset-many-tasks"),
        environment_ids=("jupyter-notebook", "python-script"),
        example_ids=("quickstart_univariate", "quickstart_dataset_pipeline", "taskify_forecasting"),
        tutorial_ids=("first-five-minutes",),
        api_names=("generate_series", "generate_series_dataset", "taskify_dataset", "build_task_context"),
        starter_id="first-success-notebook",
        title_zh="10 分钟内跑通第一次真正有意义的成功",
        summary_zh="最短引导路径：生成一条序列、解释它、创建基础数据集、任务化并保存结果。",
        promise_zh="你最后拿到的是一份真正可复用的数据资产和一个清晰心智模型，而不是一堆断开的代码片段。",
        when_to_use_zh="当你是新用户，或正在把这个包教给别人时使用。",
        steps_zh=(
            "先用 trend + seasonality + noise 生成一条序列。",
            "查看 trace 或生成一个简短 EDA 报告，理解这个库会保留什么信息。",
            "创建一个基础 SeriesDataset，并把它转成 forecasting 任务。",
            "保存 TaskDataset，让 README、schema 和 cards 跟着数据一起走。",
        ),
    ),
    Playbook(
        playbook_id="real-data-understanding",
        title="Start from real data and decide the task after the report",
        summary="Profile a real dataset first, then let tags, route suggestions, and examples narrow the modeling path.",
        promise="You avoid premature model choice and get a traceable explanation for why the first task was chosen.",
        when_to_use="Use this when you already have arrays, logs, or external rollouts and want to decide between forecasting, classification, and causal/control tasks.",
        steps=(
            "Wrap the raw series or dataset so TSDataForge can describe it.",
            "Generate a linked EDA report and read the recommended tasks and routes.",
            "Pick the nearest example and API entry point for the detected structure.",
            "Only then taskify the dataset into the first modeling protocol.",
        ),
        scenario_ids=("real-data-profile-and-taskify", "control-rollout-analysis"),
        environment_ids=("jupyter-notebook", "simulator-or-data-lake", "python-script"),
        example_ids=("real_series_eda", "dataset_eda", "real_dataset_taskify", "external_rollout_wrap"),
        tutorial_ids=("real-data-to-first-model",),
        api_names=("wrap_external_series", "describe_series", "generate_eda_report", "taskify_dataset"),
        starter_id="real-data-eda-starter",
        title_zh="从真实数据开始，在报告之后再决定任务",
        summary_zh="先对真实数据做画像，再让标签、路由建议和案例帮助你缩小建模路径。",
        promise_zh="这样可以避免过早选模型，并给“为什么先做这个任务”留下可追溯解释。",
        when_to_use_zh="当你已经有数组、日志或外部 rollout，想在 forecasting、classification、causal/control 之间做判断时使用。",
        steps_zh=(
            "先把原始序列或数据集包装进 TSDataForge。",
            "生成 linked EDA 报告，阅读推荐任务和路由建议。",
            "根据检测到的结构选择最近的案例和 API 入口。",
            "最后再把数据集任务化成第一个建模协议。",
        ),
    ),
    Playbook(
        playbook_id="benchmark-lab",
        title="Build one reusable benchmark asset, then derive several tasks",
        summary="Design a structure family once and spin out forecasting, classification, anomaly, SSL, or system-ID tasks without copying code.",
        promise="You keep the benchmark coherent while still serving several downstream evaluations.",
        when_to_use="Use this when you are building a paper benchmark, a lab-internal dataset asset, or a reusable teaching set.",
        steps=(
            "Choose the structure recipes or specs that define the benchmark family.",
            "Generate one base SeriesDataset and inspect dataset-level EDA coverage.",
            "Taskify the base asset into at least two task views.",
            "Export cards, schema, and a compact context so the benchmark is easy to share.",
        ),
        scenario_ids=("synthetic-benchmark-design", "one-base-dataset-many-tasks"),
        environment_ids=("python-script", "headless-server", "ci-pipeline"),
        example_ids=("classification_benchmark", "masked_reconstruction", "anomaly_detection", "change_point_detection"),
        tutorial_ids=("one-base-dataset-many-tasks",),
        api_names=("generate_series_dataset", "describe_dataset", "taskify_dataset", "build_dataset_context"),
        starter_id="benchmark-lab",
        title_zh="构建一份可复用 benchmark 资产，再派生多个任务",
        summary_zh="先设计结构族，再在不复制代码的情况下派生 forecasting、classification、anomaly、SSL、system-ID 等任务。",
        promise_zh="你能保持 benchmark 的整体一致性，同时服务多个下游评测。",
        when_to_use_zh="当你要做论文 benchmark、实验室内部数据资产，或一份可复用教学数据集时使用。",
        steps_zh=(
            "先选定定义 benchmark 家族的 structure recipes 或 specs。",
            "生成一份基础 SeriesDataset，并检查 dataset-level EDA 覆盖率。",
            "至少把这份基础资产任务化成两种任务视图。",
            "导出 cards、schema 和 compact context，让 benchmark 更容易分享。",
        ),
    ),
    Playbook(
        playbook_id="control-causal-evaluation",
        title="From controlled rollouts to causal and counterfactual evaluation",
        summary="Use inputs, states, interventions, and policies as first-class parts of the workflow instead of bolting them on later.",
        promise="You can compare factual and counterfactual behavior with less glue code and clearer semantics.",
        when_to_use="Use this when your time series are driven by inputs, interventions, or policies, even if they do not come from robotics.",
        steps=(
            "Start from an input-output, policy-controlled, or causal generator — or wrap an external rollout.",
            "Generate an EDA report to confirm that the dataset carries the right control or intervention signals.",
            "Taskify into system identification, causal response, intervention detection, or counterfactual response.",
            "Publish paired factual/counterfactual artifacts when comparison is the main story.",
        ),
        scenario_ids=("control-rollout-analysis", "causal-and-counterfactual-evaluation"),
        environment_ids=("python-script", "jupyter-notebook", "simulator-or-data-lake"),
        example_ids=("system_identification", "causal_response", "policy_counterfactual", "intervention_detection"),
        tutorial_ids=("control-causal-counterfactual",),
        api_names=("LinearStateSpace", "CausalVARX", "generate_counterfactual_pair", "taskify_dataset"),
        starter_id="control-causal-lab",
        title_zh="从受控 rollout 走向因果与反事实评估",
        summary_zh="把输入、状态、干预和策略作为工作流的一等公民，而不是后面再临时拼接。",
        promise_zh="这样你能用更少的胶水代码和更清晰的语义来比较 factual 与 counterfactual 行为。",
        when_to_use_zh="当你的时间序列受输入、干预或策略驱动时使用，即使它们并不来自机器人场景。",
        steps_zh=(
            "从 input-output、policy-controlled、causal 生成器开始，或者包装外部 rollout。",
            "生成 EDA 报告，确认数据集确实带有正确的 control / intervention 信号。",
            "任务化成 system identification、causal response、intervention detection 或 counterfactual response。",
            "如果核心故事是比较，就发布 factual/counterfactual 成对资产。",
        ),
    ),
    Playbook(
        playbook_id="publish-and-scale",
        title="Publish assets that humans and agents can both use",
        summary="Turn a successful workflow into a public surface with docs, examples, cards, compact contexts, and static bundles.",
        promise="You reduce support overhead while making the library easier to discover and easier to automate.",
        when_to_use="Use this when the code already works and the real bottleneck is adoption, onboarding, and reproducibility.",
        steps=(
            "Generate a static docs site with landing, tutorials, examples, API, and FAQ.",
            "Bundle linked EDA reports and dataset cards with the saved assets.",
            "Export API manifests and compact contexts so agents can stay grounded with small prompts.",
            "Publish scenario-based entry points instead of only module-based docs.",
        ),
        scenario_ids=("agent-grounded-assets", "docs-launch-and-adoption"),
        environment_ids=("static-docs-site", "ci-pipeline", "llm-agent-workflow"),
        example_ids=("agent_context_pack", "dataset_cards", "api_reference_overview", "docs_site_generation"),
        tutorial_ids=("agent-friendly-assets", "launch-and-community"),
        api_names=("generate_docs_site", "build_agent_context", "build_task_dataset_card", "save_api_reference"),
        starter_id="agent-docs-surface",
        title_zh="发布人类和 agent 都能直接使用的资产",
        summary_zh="把一个已经成功的工作流做成 docs、examples、cards、compact contexts 和静态 bundles 组成的公共表层。",
        promise_zh="这样既能降低支持成本，也能让库更容易被发现、更容易被自动化消费。",
        when_to_use_zh="当代码本身已经通了，真正瓶颈变成 adoption、onboarding 和 reproducibility 时使用。",
        steps_zh=(
            "生成静态 docs site，包含 landing、tutorials、examples、API 和 FAQ。",
            "把 linked EDA 报告和 dataset cards 一起绑定到保存好的资产里。",
            "导出 API manifests 和 compact contexts，让 agent 用更短的 prompt 也能保持 grounding。",
            "用场景入口而不是模块入口来组织公开文档。",
        ),
    ),
)


_STARTERS: tuple[StarterKit, ...] = (
    StarterKit(
        starter_id="first-success-notebook",
        title="Starter kit: first success notebook",
        summary="A tiny starter project that takes a new user from one generated series to one saved forecasting dataset.",
        best_for=("new users", "teaching", "first internal demo"),
        environment_ids=("jupyter-notebook", "python-script"),
        tutorial_ids=("first-five-minutes",),
        example_ids=("quickstart_univariate", "quickstart_dataset_pipeline", "taskify_forecasting"),
        api_names=("generate_series", "generate_series_dataset", "taskify_dataset"),
        generated_files=("README.md", "requirements.txt", "scripts/", "notebooks/", "starter_manifest.json"),
        title_zh="Starter kit：第一次成功 notebook",
        summary_zh="一个非常小的 starter project，让新用户从单条生成序列走到一个保存好的 forecasting 数据集。",
        best_for_zh=("新用户", "教学", "第一次内部演示"),
    ),
    StarterKit(
        starter_id="real-data-eda-starter",
        title="Starter kit: real data profiling and task choice",
        summary="A starter project for people who already have data and need a safe first modeling decision.",
        best_for=("analysts", "domain researchers", "dataset triage"),
        environment_ids=("jupyter-notebook", "simulator-or-data-lake"),
        tutorial_ids=("real-data-to-first-model",),
        example_ids=("real_series_eda", "dataset_eda", "real_dataset_taskify"),
        api_names=("wrap_external_series", "generate_eda_report", "taskify_dataset"),
        generated_files=("README.md", "requirements.txt", "scripts/", "notebooks/", "outputs/"),
        title_zh="Starter kit：真实数据画像与任务选择",
        summary_zh="适合已经有数据、需要一个安全的第一步建模决策的人。",
        best_for_zh=("分析师", "领域研究者", "数据集盘点"),
    ),
    StarterKit(
        starter_id="benchmark-lab",
        title="Starter kit: reusable benchmark lab",
        summary="Build one structure-aware benchmark asset and derive several task datasets from it.",
        best_for=("benchmark authors", "lab maintainers", "course projects"),
        environment_ids=("python-script", "headless-server", "ci-pipeline"),
        tutorial_ids=("one-base-dataset-many-tasks",),
        example_ids=("classification_benchmark", "masked_reconstruction", "anomaly_detection", "change_point_detection"),
        api_names=("generate_series_dataset", "taskify_dataset", "describe_dataset"),
        generated_files=("README.md", "requirements.txt", "scripts/", "notebooks/", "benchmark_assets/"),
        title_zh="Starter kit：可复用 benchmark lab",
        summary_zh="先构建一份结构感知 benchmark 资产，再从它派生多个任务数据集。",
        best_for_zh=("benchmark 作者", "实验室维护者", "课程项目"),
    ),
    StarterKit(
        starter_id="control-causal-lab",
        title="Starter kit: control, causal, and counterfactual lab",
        summary="A template for input-driven, intervention-driven, and counterfactual sequence workflows.",
        best_for=("control researchers", "causal ML", "simulator users"),
        environment_ids=("python-script", "jupyter-notebook", "simulator-or-data-lake"),
        tutorial_ids=("control-causal-counterfactual",),
        example_ids=("system_identification", "causal_response", "policy_counterfactual", "intervention_detection"),
        api_names=("LinearStateSpace", "CausalVARX", "generate_counterfactual_pair", "taskify_dataset"),
        generated_files=("README.md", "requirements.txt", "scripts/", "notebooks/", "artifacts/"),
        title_zh="Starter kit：控制、因果与反事实实验室",
        summary_zh="适合输入驱动、干预驱动和反事实序列工作流的模板。",
        best_for_zh=("控制研究者", "因果 ML", "仿真用户"),
    ),
    StarterKit(
        starter_id="agent-docs-surface",
        title="Starter kit: docs, cards, and agent-facing assets",
        summary="A starter bundle for turning a working dataset workflow into a package that can spread and survive external usage.",
        best_for=("maintainers", "platform teams", "open-source releases"),
        environment_ids=("static-docs-site", "ci-pipeline", "llm-agent-workflow"),
        tutorial_ids=("agent-friendly-assets", "launch-and-community"),
        example_ids=("agent_context_pack", "dataset_cards", "api_reference_overview", "docs_site_generation"),
        api_names=("build_agent_context", "build_task_dataset_card", "generate_docs_site", "save_api_reference"),
        generated_files=("README.md", "requirements.txt", "scripts/", "notebooks/", "site/"),
        title_zh="Starter kit：docs、cards 和 agent-facing 资产",
        summary_zh="把一个已经通了的数据工作流做成可传播、可对外使用的包表层。",
        best_for_zh=("维护者", "平台团队", "开源发布"),
    ),
)


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


def playbook_catalog(*, language: str = "en") -> list[Playbook]:
    """Return goal-first workflow playbooks for the public docs and agent surface."""
    lang = (language or "en").lower()
    if lang.startswith("zh"):
        return [
            Playbook(
                playbook_id=item.playbook_id,
                title=item.title_zh or item.title,
                summary=item.summary_zh or item.summary,
                promise=item.promise_zh or item.promise,
                when_to_use=item.when_to_use_zh or item.when_to_use,
                steps=tuple(item.steps_zh or item.steps),
                scenario_ids=item.scenario_ids,
                environment_ids=item.environment_ids,
                example_ids=item.example_ids,
                tutorial_ids=item.tutorial_ids,
                api_names=item.api_names,
                starter_id=item.starter_id,
                title_zh=item.title_zh,
                summary_zh=item.summary_zh,
                promise_zh=item.promise_zh,
                when_to_use_zh=item.when_to_use_zh,
                steps_zh=item.steps_zh,
            )
            for item in _PLAYBOOKS
        ]
    return list(_PLAYBOOKS)


def starter_catalog(*, language: str = "en") -> list[StarterKit]:
    """Return starter project templates that can be rendered into runnable bundles."""
    lang = (language or "en").lower()
    if lang.startswith("zh"):
        return [
            StarterKit(
                starter_id=item.starter_id,
                title=item.title_zh or item.title,
                summary=item.summary_zh or item.summary,
                best_for=tuple(item.best_for_zh or item.best_for),
                environment_ids=item.environment_ids,
                tutorial_ids=item.tutorial_ids,
                example_ids=item.example_ids,
                api_names=item.api_names,
                generated_files=item.generated_files,
                title_zh=item.title_zh,
                summary_zh=item.summary_zh,
                best_for_zh=item.best_for_zh,
            )
            for item in _STARTERS
        ]
    return list(_STARTERS)


def _recommend(items: list, query: str, *, top_k: int = 3):
    q = _norm_tokens(query)
    scored = []
    for i, item in enumerate(items):
        tokens = set()
        for key, value in item.to_dict(language="en").items():
            if isinstance(value, (str, tuple, list)):
                tokens |= _norm_tokens(value)
        overlap = len(q & tokens)
        score = 3.0 * overlap
        title = getattr(item, "title", "")
        summary = getattr(item, "summary", "")
        score += 1.5 * sum(1 for tok in q if tok in _norm_tokens(title))
        score += 0.5 * sum(1 for tok in q if tok in _norm_tokens(summary))
        scored.append((score, -i, item))
    scored.sort(reverse=True)
    return [item for _, _, item in scored[:top_k]]


def recommend_playbooks(query: str, *, top_k: int = 3, language: str = "en") -> list[Playbook]:
    """Recommend playbooks for a natural-language goal or onboarding need."""
    return _recommend(playbook_catalog(language=language), query, top_k=top_k)


def recommend_starters(query: str, *, top_k: int = 3, language: str = "en") -> list[StarterKit]:
    """Recommend starter kits for a natural-language use case or environment."""
    return _recommend(starter_catalog(language=language), query, top_k=top_k)


def _markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip() + "\n",
    }


def _code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": code.rstrip() + "\n",
    }


def _lookup_examples() -> dict[str, ExampleRecipe]:
    return {ex.example_id: ex for ex in example_catalog(language="en")}


def _lookup_tutorials(language: str) -> dict[str, TutorialTrack]:
    return {item.tutorial_id: item for item in tutorial_catalog(language=language)}


def _notebook_dict(track: TutorialTrack, *, language: str = "en") -> dict:
    examples = _lookup_examples()
    title = track.title if not language.startswith("zh") else (track.title_zh or track.title)
    summary = track.summary if not language.startswith("zh") else (track.summary_zh or track.summary)
    steps = track.steps if not language.startswith("zh") else (track.steps_zh or track.steps)
    outcomes = track.outcomes if not language.startswith("zh") else (track.outcomes_zh or track.outcomes)
    cells: list[dict] = [
        _markdown_cell(
            f"# {title}\n\n{summary}\n\n"
            + "## Outcomes\n\n"
            + "\n".join([f"- {item}" for item in outcomes])
            + "\n\n## Steps\n\n"
            + "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
        )
    ]
    cells.append(
        _markdown_cell(
            "## Imports\n\nRun the cells below in order. Each section is based on a tested example recipe from the docs surface."
            if not language.startswith("zh")
            else "## 导入\n\n按顺序运行下面的单元。每个部分都来自文档中已经测试过的 example recipe。"
        )
    )
    for ex_id in track.example_ids:
        ex = examples.get(ex_id)
        if not ex:
            continue
        title_ex = ex.title
        summary_ex = ex.summary
        cells.append(_markdown_cell(f"## {title_ex}\n\n{summary_ex}"))
        cells.append(_code_cell(ex.code))
    cells.append(
        _markdown_cell(
            "## Next moves\n\n- Save the asset you want to keep.\n- Generate an EDA report if you need explanation.\n- Export a card or context if the result must travel to other people or agents."
            if not language.startswith("zh")
            else "## 下一步\n\n- 把你要保留的资产保存下来。\n- 如果需要解释，生成 EDA 报告。\n- 如果结果要交给他人或 agent，再导出 card 或 context。"
        )
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
            "tsdataforge": {
                "tutorial_id": track.tutorial_id,
                "language": language,
                "example_ids": list(track.example_ids),
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def export_tutorial_notebooks(output_dir: str | Path, *, tutorial_ids: Iterable[str] | None = None, language: str = "en") -> list[NotebookAsset]:
    """Export tutorial tracks as `.ipynb` and `.py` assets backed by example recipes."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tutorials = _lookup_tutorials(language)
    selected_ids = list(tutorial_ids) if tutorial_ids is not None else list(tutorials.keys())
    assets: list[NotebookAsset] = []
    manifest: list[dict] = []
    for tutorial_id in selected_ids:
        track = tutorials.get(tutorial_id)
        if track is None:
            continue
        notebook = _notebook_dict(track, language=language)
        ipynb_path = out / f"{tutorial_id}.ipynb"
        py_path = out / f"{tutorial_id}.py"
        ipynb_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
        py_parts = [f"# {track.title if not language.startswith('zh') else (track.title_zh or track.title)}\n"]
        for ex_id in track.example_ids:
            ex = _lookup_examples().get(ex_id)
            if not ex:
                continue
            py_parts.append(f"\n# --- {ex.title} ---\n")
            py_parts.append(ex.code.rstrip() + "\n")
        py_path.write_text("\n".join(py_parts).strip() + "\n", encoding="utf-8")
        asset = NotebookAsset(
            notebook_id=tutorial_id,
            title=track.title if not language.startswith("zh") else (track.title_zh or track.title),
            summary=track.summary if not language.startswith("zh") else (track.summary_zh or track.summary),
            tutorial_id=tutorial_id,
            path_ipynb=str(ipynb_path),
            path_py=str(py_path),
        )
        assets.append(asset)
        manifest.append(asset.to_dict())
    (out / "notebook_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return assets


def _starter_readme(starter: StarterKit, *, language: str = "en") -> str:
    tutorials = _lookup_tutorials(language)
    title = starter.title if not language.startswith("zh") else (starter.title_zh or starter.title)
    summary = starter.summary if not language.startswith("zh") else (starter.summary_zh or starter.summary)
    best_for = starter.best_for if not language.startswith("zh") else (starter.best_for_zh or starter.best_for)
    lines = [f"# {title}", "", summary, "", "## Best for", ""]
    lines.extend([f"- {item}" for item in best_for])
    lines.extend([
        "",
        "## What this starter contains" if not language.startswith("zh") else "## 这个 starter 包含什么",
        "",
        "- README.md",
        "- requirements.txt",
        "- scripts/",
        "- notebooks/",
        "- starter_manifest.json",
        "",
        "## Tutorial tracks included" if not language.startswith("zh") else "## 包含的 tutorial 路径",
        "",
    ])
    for tutorial_id in starter.tutorial_ids:
        item = tutorials.get(tutorial_id)
        if item is None:
            continue
        t_title = item.title if not language.startswith("zh") else (item.title_zh or item.title)
        lines.append(f"- `{tutorial_id}` — {t_title}")
    lines.extend([
        "",
        "## How to use it" if not language.startswith("zh") else "## 如何使用",
        "",
        "1. Install the package and optional visualization dependencies." if not language.startswith("zh") else "1. 安装包本体和可选的可视化依赖。",
        "2. Run the scripts in order or open the notebooks first." if not language.startswith("zh") else "2. 按顺序运行 scripts，或者先打开 notebooks。",
        "3. Save the outputs you want to keep as reusable assets." if not language.startswith("zh") else "3. 把你想保留的输出保存成可复用资产。",
        "",
    ])
    return "\n".join(lines).strip() + "\n"


def create_starter_project(output_dir: str | Path, starter_id: str, *, language: str = "en") -> StarterProjectResult:
    """Render one starter kit into a small runnable project directory."""
    starter = {item.starter_id: item for item in starter_catalog(language=language)}.get(starter_id)
    if starter is None:
        raise KeyError(f"Unknown starter_id: {starter_id}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scripts_dir = out / "scripts"
    notebooks_dir = out / "notebooks"
    outputs_dir = out / "outputs"
    for d in (scripts_dir, notebooks_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    readme_path = out / "README.md"
    readme_path.write_text(_starter_readme(starter, language=language), encoding="utf-8")
    requirements = [
        "tsdataforge>=0.3.4",
        "matplotlib>=3.7  # optional but recommended for HTML EDA reports",
    ]
    (out / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")

    examples = {item.example_id: item for item in example_catalog(language=language)}
    script_paths: list[str] = []
    for idx, ex_id in enumerate(starter.example_ids, start=1):
        ex = examples.get(ex_id)
        if ex is None:
            continue
        path = scripts_dir / f"{idx:02d}_{ex_id}.py"
        path.write_text(
            dedent(
                f"""\
                \"\"\"{ex.title}\n\n                {ex.summary}\n                \"\"\"\n\n                {ex.code.rstrip()}\n                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        script_paths.append(str(path))

    notebook_assets = export_tutorial_notebooks(notebooks_dir, tutorial_ids=starter.tutorial_ids, language=language)
    notebook_paths = tuple([asset.path_ipynb for asset in notebook_assets] + [asset.path_py for asset in notebook_assets])

    manifest = {
        "starter_id": starter.starter_id,
        "title": starter.title,
        "summary": starter.summary,
        "best_for": list(starter.best_for),
        "environment_ids": list(starter.environment_ids),
        "tutorial_ids": list(starter.tutorial_ids),
        "example_ids": list(starter.example_ids),
        "api_names": list(starter.api_names),
        "generated_files": starter.generated_files,
        "script_paths": script_paths,
        "notebook_paths": list(notebook_paths),
    }
    manifest_path = out / "starter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return StarterProjectResult(
        starter_id=starter.starter_id,
        output_dir=str(out),
        manifest_path=str(manifest_path),
        readme_path=str(readme_path),
        script_paths=tuple(script_paths),
        notebook_paths=notebook_paths,
    )



_PLAYBOOKS = _PLAYBOOKS + (
    Playbook(
        playbook_id="traffic-case-studies",
        title="Turn current public-interest signals into explainable time-series case studies",
        summary="Use live GitHub, crypto, and commodity series to publish case studies that both attract attention and teach the package.",
        promise="You get a reproducible public-data story, an EDA report, and a similarity workflow instead of a one-off social-media chart.",
        when_to_use="Use this when you want examples with real attention value over the next few months, not only synthetic demos.",
        steps=(
            "Pick a live public signal family such as GitHub stars, crypto, or macro/commodity prices.",
            "Fetch the series with one of the live integrations and generate an EDA report first.",
            "Compare the shapes with pairwise_similarity or find_top_matches depending on the story you want to tell.",
            "Export the report, compact context, and docs bundle so the case study is easy to share and rerun.",
        ),
        scenario_ids=("live-market-and-attention-analysis", "docs-launch-and-adoption"),
        environment_ids=("jupyter-notebook", "python-script", "static-docs-site", "llm-agent-workflow"),
        example_ids=("openclaw_stars_similarity", "github_stars_pairwise_panel", "btc_gold_oil_similarity"),
        tutorial_ids=("live-signals-and-similarity",),
        api_names=("fetch_github_stars_series", "fetch_coingecko_market_chart", "fetch_fred_series", "pairwise_similarity", "find_top_matches", "generate_eda_report"),
        starter_id="live-case-study-lab",
        title_zh="把当前公共热点信号做成可解释的时间序列案例",
        summary_zh="用 GitHub、加密货币和大宗商品等实时序列发布既有传播力、又能教学的案例。",
        promise_zh="你得到的是一套可复现的公共数据故事、EDA 报告和相似性工作流，而不是一次性的社交媒体图表。",
        when_to_use_zh="当你希望未来几个月都有真实关注度的案例，而不是只有合成 demo 时使用。",
        steps_zh=(
            "先选一类公共信号：GitHub stars、加密货币，或宏观/大宗商品价格。",
            "用 live integration 抓取序列，并先生成 EDA 报告。",
            "根据你要讲的故事，选择 pairwise_similarity 或 find_top_matches 做形状比较。",
            "导出报告、compact context 和 docs bundle，让案例易于分享和复跑。",
        ),
    ),
)

_STARTERS = _STARTERS + (
    StarterKit(
        starter_id="live-case-study-lab",
        title="Live case-study lab",
        summary="A starter project for public GitHub/market case studies with live fetchers, similarity analysis, and a shareable report pipeline.",
        best_for=("public-interest demos", "research outreach", "market or attention analysis", "shareable examples"),
        environment_ids=("jupyter-notebook", "python-script", "static-docs-site"),
        tutorial_ids=("live-signals-and-similarity",),
        example_ids=("openclaw_stars_similarity", "github_stars_pairwise_panel", "btc_gold_oil_similarity"),
        api_names=("fetch_github_stars_series", "fetch_coingecko_market_chart", "fetch_fred_series", "pairwise_similarity", "find_top_matches", "generate_eda_report"),
        generated_files=("README.md", "requirements.txt", "scripts/", "notebooks/", "starter_manifest.json"),
        title_zh="实时案例实验室",
        summary_zh="一个面向 GitHub/市场公共案例的 starter project，内置 live fetcher、相似性分析和可分享报告流程。",
        best_for_zh=("公共热点 demo", "研究传播", "市场或关注度分析", "可分享案例"),
    ),
)
