from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from html import escape
from pathlib import Path
from textwrap import dedent
from typing import Iterable

from .api_reference import APIReference, APICategory, APISymbol, build_api_reference
from .eda_linking import FAQ_ENTRIES, common_eda_finding_routes, example_eda_routes, api_eda_routes
from .examples import ExampleRecipe, example_catalog, examples_by_category, recommend_examples
from .tutorials import TutorialTrack, tutorial_catalog, recommend_tutorials
from .playbooks import Playbook, StarterKit, playbook_catalog, starter_catalog, recommend_playbooks, recommend_starters, export_tutorial_notebooks, create_starter_project
from .scenarios import EnvironmentProfile, ScenarioProfile, environment_catalog, scenario_catalog, recommend_scenarios, recommend_environments
from .positioning import PackageProfile, PositioningMatrix, build_positioning_matrix, competitor_catalog, recommend_companions


STYLE = dedent(
    """
    :root {
      --bg: #ffffff;
      --bg-muted: #f5f7fa;
      --panel: #ffffff;
      --panel-muted: #f8fafc;
      --text: #16202a;
      --muted: #5c6775;
      --accent: #f89939;
      --accent-soft: #fff3e5;
      --link: #1f5fbf;
      --border: #d9dee5;
      --border-strong: #c7cfd9;
      --shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
      --code-bg: #f7f8fa;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: "Segoe UI", Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    a { color: var(--link); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .nav {
      position: sticky; top: 0; z-index: 20;
      background: rgba(255,255,255,0.96);
      border-bottom: 1px solid var(--border);
    }
    .nav-inner {
      max-width: 1380px;
      margin: 0 auto;
      padding: 12px 20px;
      display: flex;
      gap: 14px;
      align-items: center;
      justify-content: space-between;
    }
    .brand {
      color: var(--text);
      font-weight: 700;
      letter-spacing: 0.01em;
      font-size: 18px;
    }
    .brand small {
      color: var(--muted);
      font-weight: 600;
      margin-left: 8px;
    }
    .nav-right { display: flex; gap: 10px; align-items: center; }
    .top-link {
      display: inline-flex;
      align-items: center;
      padding: 7px 11px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      font-size: 14px;
    }
    .lang-switch {
      display: inline-flex;
      align-items: center;
      padding: 7px 12px;
      border-radius: 6px;
      background: var(--panel);
      border: 1px solid var(--border);
      color: var(--text);
      font-size: 14px;
    }
    .shell {
      max-width: 1380px;
      margin: 0 auto;
      padding: 24px 20px 64px;
    }
    .layout {
      display: grid;
      grid-template-columns: 250px minmax(0, 1fr);
      gap: 28px;
      align-items: start;
    }
    .sidebar {
      position: sticky;
      top: 76px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--panel);
      padding: 18px 16px;
      box-shadow: var(--shadow);
    }
    .sidebar-title {
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 12px;
      font-weight: 700;
    }
    .sidebar-nav {
      display: grid;
      gap: 4px;
    }
    .sidebar-link {
      display: block;
      padding: 8px 10px;
      border-radius: 6px;
      color: var(--text);
      font-size: 14px;
    }
    .sidebar-link.active {
      background: var(--accent-soft);
      border-left: 3px solid var(--accent);
      font-weight: 700;
      padding-left: 7px;
    }
    .main {
      min-width: 0;
    }
    .hero {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 28px;
      background: var(--panel-muted);
      box-shadow: var(--shadow);
    }
    .hero h1 {
      margin: 0 0 10px 0;
      font-size: 38px;
      line-height: 1.12;
    }
    .hero p { color: var(--muted); font-size: 17px; max-width: 900px; margin: 0; }
    .hero-meta {
      margin-top: 16px;
      display: grid;
      gap: 10px;
    }
    .hero-meta-line {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 14px;
    }
    .hero-meta-line img {
      vertical-align: middle;
    }
    .hero-meta small {
      color: var(--muted);
    }
    .badge-links {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .badge-links a {
      display: inline-flex;
      align-items: center;
    }
    .badges, .pills, .toc { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
    .badge, .pill, .toc a {
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      font-size: 13px;
      background: var(--panel);
      color: var(--muted);
    }
    .badge { border-color: #f0c18c; background: #fff8ef; color: #8a4d00; }
    .pill { background: #f8fafc; color: var(--muted); }
    .toc {
      padding: 12px 14px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--panel);
    }
    .toc a { border-radius: 6px; }
    .section { margin-top: 34px; }
    .section h2 { font-size: 28px; margin-bottom: 10px; line-height: 1.2; }
    .section h3 { font-size: 21px; margin-bottom: 8px; line-height: 1.3; }
    .kicker { color: var(--muted); text-transform: uppercase; font-weight: 700; font-size: 12px; letter-spacing: 0.08em; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin-top: 16px; }
    .grid.tight { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
    .two-col { display: grid; grid-template-columns: 1.12fr 0.88fr; gap: 18px; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 18px;
      box-shadow: var(--shadow);
    }
    .card p, .card li { color: var(--muted); }
    .card h3, .card h4 { margin-top: 0; }
    .table { width: 100%; border-collapse: collapse; margin-top: 10px; background: var(--panel); border: 1px solid var(--border); }
    .table th, .table td { border-bottom: 1px solid var(--border); padding: 11px 10px; text-align: left; vertical-align: top; }
    .table th { color: var(--text); background: var(--panel-muted); font-weight: 700; }
    .small { font-size: 13px; color: var(--muted); }
    .muted { color: var(--muted); }
    .tip, .notice, .warning {
      border-radius: 8px;
      padding: 12px 14px;
      border: 1px solid var(--border);
      border-left: 4px solid #2b7a4b;
      background: #f4faf6;
    }
    .notice { border-left-color: var(--accent); background: #fff8ef; }
    .warning { border-left-color: #d97706; background: #fff7e8; }
    .breadcrumbs { color: var(--muted); margin: 12px 0 16px; font-size: 14px; }
    .breadcrumbs a { color: var(--muted); }
    pre {
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      overflow: auto;
      color: #0f172a;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: rgba(15, 23, 42, 0.04);
      border-radius: 4px;
      padding: 0 4px;
    }
    pre code {
      background: transparent;
      padding: 0;
    }
    .footer {
      margin-top: 56px;
      color: var(--muted);
      font-size: 13px;
      border-top: 1px solid var(--border);
      padding-top: 18px;
    }
    .link-list { padding-left: 18px; }
    .link-list li { margin-bottom: 8px; }
    @media (max-width: 1080px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; }
      .two-col { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      .nav-inner { padding: 12px 14px; }
      .shell { padding: 18px 14px 48px; }
      .hero { padding: 22px; }
      .hero h1 { font-size: 30px; }
    }
    """
).strip()


@dataclass
class DocsSiteResult:
    output_dir: str
    pages: list[str] = field(default_factory=list)
    example_pages: list[str] = field(default_factory=list)
    api_pages: list[str] = field(default_factory=list)
    search_index: str | None = None
    api_manifest: str | None = None
    faq_page: str | None = None
    eda_route_map: str | None = None
    zh_pages: list[str] = field(default_factory=list)
    notebook_files: list[str] = field(default_factory=list)
    starter_projects: list[str] = field(default_factory=list)
    showcase_bundles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _tr(lang: str, en: str, zh: str) -> str:
    return zh if lang.startswith("zh") else en


NAV_ITEMS = [
    ("index.html", "Docs", "文档首页"),
    ("getting-started.html", "Quickstart", "快速上手"),
    ("handoff.html", "Handoff", "交接"),
    ("showcase.html", "Showcase", "案例展示"),
    ("tutorials.html", "Tutorials", "教程"),
    ("playbooks.html", "Workflows", "工作流"),
    ("starter-kits.html", "Starter Projects", "起步项目"),
    ("positioning.html", "Positioning", "生态定位"),
    ("cookbook.html", "Examples", "案例库"),
    ("taskification.html", "Taskification", "任务化"),
    ("agent-playbook.html", "Agent Guide", "Agent 指南"),
    ("use-cases.html", "Use Cases", "应用场景"),
    ("rollout.html", "Release", "发布与传播"),
    ("api-reference.html", "API", "API"),
    ("faq.html", "FAQ", "FAQ"),
]

CATEGORY_LABELS = {
    "quickstart": ("Quickstart", "快速上手"),
    "taskification": ("Taskification", "任务化"),
    "eda": ("Real-data EDA", "真实数据 EDA"),
    "control_causal": ("Control / causal / counterfactual", "控制 / 因果 / 反事实"),
    "integration": ("Integrations", "外部接入"),
    "agent": ("Agent & assets", "Agent 与资产"),
    "launch": ("Release & docs", "发布 / 文档 / 社区"),
    "live_similarity": ("Live popularity & market similarity", "热点关注度与市场相似性"),
}

API_CATEGORY_LABELS = {
    "workflow": ("Workflow entry points", "工作流入口"),
    "containers": ("Core containers and specs", "数据容器与规范"),
    "analysis": ("Description, explanation, and EDA", "描述、解释与 EDA"),
    "agent": ("Agent, docs, and launch surface", "Agent / 文档 / 扩散接口"),
    "causal_control": ("Control, causal, and counterfactual", "控制、因果与反事实"),
    "operators": ("Composition and observation layer", "组合算子与观测机制"),
    "primitives": ("Primitive generators and dynamics", "基础原语与动态组件"),
}

API_CATEGORY_SUMMARIES = {
    "workflow": (
        "The most common paths begin here: generate a series, generate a base dataset, taskify it, or generate a task dataset directly.",
        "最常见的路径都从这里开始：生成序列、生成基础数据集、任务化，或直接生成任务数据集。",
    ),
    "containers": (
        "These are the core objects that make the library composable, serializable, and agent-friendly.",
        "这些是让库可组合、可序列化、对 agent 友好的核心对象。",
    ),
    "analysis": (
        "Describe first, model second. These APIs are how you understand real data before committing to a task.",
        "先描述，再建模。这些接口帮助你在选任务前理解真实数据。",
    ),
    "agent": (
        "Compact contexts, cards, docs generation, and manifests that keep agent workflows low-token and grounded.",
        "compact context、card、docs 生成和 manifest，让 agent 工作流更省 token、更稳。",
    ),
    "causal_control": (
        "State, input, output, intervention, policy, and counterfactual interfaces live here.",
        "state、input、output、intervention、policy 与 counterfactual 接口都在这里。",
    ),
    "operators": (
        "Composition operators and observation controls let you separate latent structure from measurement conditions.",
        "组合算子与观测层控制让你把 latent structure 和 measurement conditions 分开处理。",
    ),
    "primitives": (
        "These are the structural building blocks: trend, seasonality, noise, events, and dynamics.",
        "这些是结构原语：trend、seasonality、noise、events 和 dynamics。",
    ),
}


FAQ_ANSWERS_ZH = {
    "eda_links_how": "这些链接不是给某一份报告硬编码的，而是由结构标签、采样/缺失特征、trace 线索以及一份共享 route map 共同生成。这同一份 mapping 会同时驱动报告、案例页、API 页和 FAQ 页。",
    "eda_after_report": "先看报告里的 recommended tasks，再结合你的科学问题或产品目标筛选。趋势和季节性通常更适合 forecasting；多变量耦合更适合 classification 或 system identification；干预、图结构或 trace 线索更适合 causal 或 counterfactual 任务。",
    "missing_irregular_next": "优先把 observation mechanism 显式建模。之后再决定是做 masked reconstruction、robust forecasting，还是 regular-vs-irregular 的 synthetic 对照。不要指望下游模型默默吸收这些问题。",
    "taskify_vs_generate": "如果任务已经明确，而且你只想最快拿到 X/y，就直接用 generate_dataset。若想复用同一份基础数据到多个任务，或先做 EDA，则建议先 generate_series_dataset，再 taskify。",
    "real_data_no_trace": "可以。真实数据仍然可以 describe、生成报告并 taskify。只有像 causal discovery 的已知 adjacency 或 ITE 监督这类需要显式真值的任务，通常才需要合成数据或外部标注。",
    "compact_context_why": "因为 agent 默认不应该先读完整 README 和原始数组。compact context、card 和 schema 能把关键语义压进更小、更稳定的 token 预算里。",
    "simulator_scope": "不会。外部仿真或真实 rollout 一般通过 wrap_external_series 或 SeriesDataset.from_arrays 接入。TSDataForge 重点放在 taxonomy、EDA、taskification 和 agent-friendly assets。",
    "taskify_semantics": "不会。TaskDataset.schema 会把 X、y、masks 和 aux 的语义写清楚，而基础 SeriesDataset 仍然保留 values、time、meta 和 trace。",
    "public_surface": "优先公开 landing page、五分钟 quickstart、10–20 个可复制案例、API reference、FAQ，以及少量带 card 与 compact context 的样例资产。",
    "agent_minimal_stack": "大多数情况下，build_agent_context、build_*_card、API manifest、recommend_examples 和 TaskDataset.schema 就足够构成一个稳定的一阶集成。",
}

FAQ_TRANSLATIONS = {
    "eda_links_how": ("How are the landing / docs / examples / API / FAQ links inside an EDA report generated?", "EDA 报告中的 landing / docs / examples / API / FAQ 链接是怎么来的？"),
    "eda_after_report": ("After a real-data EDA report, how do I choose between forecasting, classification, and causal tasks?", "看完真实数据 EDA 报告后，下一步是 forecasting、classification 还是 causal？"),
    "missing_irregular_next": ("What should I do when the report flags missingness or irregular sampling?", "如果报告提示 missing 或 irregular_sampling，下一步该做什么？"),
    "taskify_vs_generate": ("When should I call generate_dataset directly vs generate_series_dataset + taskify?", "什么时候直接用 generate_dataset，什么时候先 generate_series_dataset 再 taskify？"),
    "real_data_no_trace": ("Can real data still be useful without trace ground truth?", "真实数据没有 trace，还能用吗？"),
    "compact_context_why": ("Why emphasize compact contexts and artifact cards?", "为什么要强调 compact context 和 card？"),
    "simulator_scope": ("Will TSDataForge rebuild existing simulation stacks?", "TSDataForge 会不会重复造仿真轮子？"),
    "taskify_semantics": ("Does taskification destroy raw-data semantics?", "任务化会不会把原始语义搞丢？"),
    "public_surface": ("What should I publish first for the public surface?", "最适合先公开哪些内容？"),
    "agent_minimal_stack": ("What is the minimum stack for an agent pipeline?", "如果要接入 agent pipeline，最小可用集是什么？"),
}

ROUTE_ZH = {
    "trend_seasonal": ("趋势 / 周期主导的真实序列", "当报告提示 trend、seasonal 或 dominant_period 明显时，通常应该先走 forecasting / decomposition / spec matching。"),
    "missing_irregular": ("缺失 / 不规则采样优先处理", "当报告提示 missing 或 irregular_sampling，优先把观测机制显式化，再决定是 masked reconstruction、鲁棒预测还是 synthetic 对照。"),
    "multivariate_coupled": ("多变量 / 耦合结构", "多通道且跨通道相关明显时，通常应该同时考虑 classification、system identification 和 causal_response。"),
    "bursty_events": ("bursty / spikes / event-like 结构", "尖峰、重尾或稀疏事件明显时，异常检测、event detection 和干预检测通常更有意义。"),
    "regime_change": ("regime / changepoint / switching 行为", "均值、方差或动态规则明显切换时，先考虑变化点任务和 regime-aware benchmark。"),
    "control_policy": ("控制 / 输入输出 / 策略驱动序列", "如果报告来自带 input/state/output/reward 的 rollout，应优先考虑 system identification、policy value 和 counterfactual。"),
    "causal_intervention": ("因果 / 干预 / 反事实线索", "当 trace 或数据 schema 包含 treatment、adjacency、counterfactual 或 intervention mask 时，优先走 causal_response / discovery / ITE / counterfactual。"),
    "dataset_inventory": ("数据集盘点 / 覆盖率 / 结构分桶", "面对的是一整批数据时，先做 dataset-level EDA、signature 统计和 task routing。"),
    "agent_surface": ("agent 友好的数据资产", "当报告已经可以稳定描述数据时，下一步通常是生成 compact context、card 和 schema 驱动的资产。"),
}

TOP_API_SUMMARIES_EN = {
    "generate_series": "Compile a spec or a list of components into one GeneratedSeries.",
    "generate_series_dataset": "Generate a reusable base SeriesDataset instead of a single task-specific X/y pair.",
    "taskify_dataset": "Convert a SeriesDataset into a task-specific TaskDataset with explicit schema.",
    "generate_dataset": "Generate a task dataset in one step when the task is already known.",
    "describe_series": "Describe a single series with tags, scores, and interpretable structural hints.",
    "describe_dataset": "Describe a whole dataset with tag counts, signatures, and distribution summaries.",
    "generate_eda_report": "Render a linked HTML EDA report for one series.",
    "generate_dataset_eda_report": "Render a linked HTML EDA report for a dataset.",
    "build_agent_context": "Compress a series, dataset, or task into a low-token agent-ready context.",
    "build_dataset_context": "Build a compact context for a base dataset.",
    "build_task_context": "Build a compact context for a task dataset.",
    "build_api_reference": "Build a structured inventory of the public API surface.",
    "save_api_reference": "Write the API reference to JSON or Markdown.",
    "generate_docs_site": "Render a static docs site with tutorials, examples, API pages, and FAQ.",
}


def _page_title(lang: str, slug: str) -> str:
    for href, en, zh in NAV_ITEMS:
        if href == slug:
            return _tr(lang, en, zh)
    return slug



def _switch_href(lang: str, slug: str, level: int = 0) -> str:
    if lang.startswith("zh"):
        return f"{'../' * (level + 1)}{slug}"
    return f"{'../' * level}zh/{slug}"



def _nav_html(lang: str, *, base: str = "", current_slug: str = "") -> str:
    links = []
    for href, en, zh in NAV_ITEMS:
        active = " active" if href == current_slug else ""
        links.append(f"<a class='sidebar-link{active}' href='{base}{href}'>{escape(_tr(lang, en, zh))}</a>")
    return "".join(links)



def _page(title: str, body: str, *, lang: str, slug: str, base: str = "", level: int = 0) -> str:
    switch_label = _tr(lang, "中文", "English")
    return f"""<!doctype html>
<html lang='{escape(lang)}'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>{escape(title)}</title>
  <style>{STYLE}</style>
</head>
<body>
  <div class='nav'>
    <div class='nav-inner'>
      <div class='brand'>TSDataForge <small>Docs v0.3.7</small></div>
      <div class='nav-right'>
        <a class='top-link' href='{base}index.html'>{escape(_tr(lang, 'Docs home', '文档首页'))}</a>
        <a class='top-link' href='https://github.com/ZipengWu365/TSDataForge'>GitHub</a>
        <a class='lang-switch' href='{_switch_href(lang, slug, level=level)}'>{escape(switch_label)}</a>
      </div>
    </div>
  </div>
  <div class='shell'>
    <div class='layout'>
      <aside class='sidebar'>
        <div class='sidebar-title'>{escape(_tr(lang, 'Documentation', '文档导航'))}</div>
        <div class='sidebar-nav'>{_nav_html(lang, base=base, current_slug=slug)}</div>
      </aside>
      <main class='main'>
        {body}
        <div class='footer'>Generated by <code>tsdataforge.agent.generate_docs_site</code>. {_tr(lang, 'This is a fully static bilingual HTML bundle.', '这是一个完全静态的双语 HTML bundle。')}</div>
      </main>
    </div>
  </div>
</body>
</html>"""



def _hero(title: str, subtitle: str, badges: Iterable[str], *, pills: Iterable[str] = ()) -> str:
    badge_html = ''.join([f"<span class='badge'>{escape(x)}</span>" for x in badges])
    pill_html = ''.join([f"<span class='pill'>{escape(x)}</span>" for x in pills])
    return f"<div class='hero'><h1>{escape(title)}</h1><p>{escape(subtitle)}</p><div class='badges'>{badge_html}</div><div class='pills'>{pill_html}</div></div>"


def _home_identity_strip(lang: str) -> str:
    docs_label = _tr(lang, "Docs", "Docs")
    repo_label = _tr(lang, "Repo", "Repo")
    license_label = _tr(lang, "MIT", "MIT")
    python_label = _tr(lang, "Python 3.10+", "Python 3.10+")
    built_text = _tr(
        lang,
        "Built by Zipeng Wu (zxw365@student.bham.ac.uk) at The University of Birmingham.",
        "由 Zipeng Wu（zxw365@student.bham.ac.uk）构建，来自 The University of Birmingham。",
    )
    return (
        "<div class='hero-meta'>"
        "<div class='badge-links'>"
        "<a href='https://zipengwu365.github.io/TSDataForge/'><img alt='Docs' src='https://img.shields.io/badge/docs-GitHub%20Pages-0b57d0'></a>"
        "<a href='https://github.com/ZipengWu365/TSDataForge'><img alt='Repo' src='https://img.shields.io/badge/repo-TSDataForge-111827'></a>"
        "<a href='https://github.com/ZipengWu365/TSDataForge/blob/main/LICENSE'><img alt='License: MIT' src='https://img.shields.io/badge/license-MIT-16a34a'></a>"
        "<span title='" + escape(python_label) + "'><img alt='Python 3.10+' src='https://img.shields.io/badge/python-3.10%2B-2563eb'></span>"
        "</div>"
        "<div class='hero-meta-line'>"
        "<a href='https://www.birmingham.ac.uk/'>"
        "<img src='https://www.birmingham.ac.uk/_s1t51Q_30649ea7-7b67-4dd9-9d18-4db7fd5c8933/static/img/icons/favicon-32x32.png' alt='University of Birmingham' width='18' height='18'>"
        "</a>"
        f"<span>{escape(built_text)}</span>"
        "</div>"
        "<div class='hero-meta-line'>"
        f"<small><a href='https://zipengwu365.github.io/TSDataForge/'>{escape(docs_label)}</a> · <a href='https://github.com/ZipengWu365/TSDataForge'>{escape(repo_label)}</a> · <a href='mailto:zxw365@student.bham.ac.uk'>zxw365@student.bham.ac.uk</a> · <a href='https://github.com/ZipengWu365/TSDataForge/blob/main/LICENSE'>{escape(license_label)}</a></small>"
        "</div>"
        "</div>"
    )



def _card(title: str, body: str) -> str:
    return f"<div class='card'><h3>{escape(title)}</h3>{body}</div>"



def _bullets(items: Iterable[str]) -> str:
    return "<ul>" + ''.join([f"<li>{escape(item)}</li>" for item in items]) + "</ul>"



def _ordered_list(items: Iterable[str]) -> str:
    return "<ol>" + ''.join([f"<li>{escape(item)}</li>" for item in items]) + "</ol>"



def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = ''.join([f"<th>{escape(h)}</th>" for h in headers])
    body = ''.join(["<tr>" + ''.join([f"<td>{cell}</td>" for cell in row]) + "</tr>" for row in rows])
    return f"<table class='table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"



def _toc(items: list[tuple[str, str]]) -> str:
    return "<div class='toc'>" + ''.join([f"<a href='#{escape(anchor)}'>{escape(label)}</a>" for anchor, label in items]) + "</div>"



def _bring_your_own_data_section(lang: str, *, section_id: str = "your-data") -> str:
    saved_file_code = """from tsdataforge import handoff

bundle = handoff(
    'my_sensor_windows.npy',
    output_dir='sensor_bundle',
    dataset_id='pump_lab_run',
)
print(bundle.output_dir)"""
    dataframe_code = """import pandas as pd
from tsdataforge import handoff

df = pd.read_csv('pump_run.csv')
values = df[['temperature', 'pressure']].to_numpy()
time = df['seconds'].to_numpy()

bundle = handoff(
    values,
    time=time,
    output_dir='pump_bundle',
    dataset_id='pump_run',
    channel_names=['temperature', 'pressure'],
)"""
    rows = [
        [
            "<code>(length,)</code>",
            escape(_tr(lang, "one univariate series", "一条单变量序列")),
        ],
        [
            "<code>(n_series, length)</code>",
            escape(_tr(lang, "one row per series", "每一行代表一条序列")),
        ],
        [
            "<code>(length, n_channels)</code> + <code>time.shape == (length,)</code>",
            escape(_tr(lang, "one multichannel series over time", "一条带多个通道的时间序列")),
        ],
        [
            "<code>(n_series, length, n_channels)</code>",
            escape(_tr(lang, "many multichannel series", "多条多通道序列")),
        ],
    ]
    body = f"<div class='section' id='{escape(section_id)}'><div class='kicker'>{escape(_tr(lang, 'Bring your own data', '接入你自己的数据'))}</div>"
    body += f"<h2>{escape(_tr(lang, 'Use your own files or arrays without guessing the input rules', '直接用你自己的文件或数组，不用猜输入规则'))}</h2>"
    body += (
        "<div class='notice'><strong>"
        + escape(_tr(lang, 'Important', '重要'))
        + ":</strong> "
        + escape(_tr(lang, 'Direct CSV/TXT loading expects numeric files. If your file has headers or date strings, read it yourself first and pass `values` plus `time=`.', '直接读取 CSV/TXT 时要求文件是纯数值。如果文件里有表头或日期字符串，请先自己读入，再传 `values` 和 `time=`。'))
        + "</div>"
    )
    body += "<div class='grid'>"
    body += _card(
        _tr(lang, '1) Start from one saved file', '1）从一个已保存文件开始'),
        "<pre><code>"
        + escape(saved_file_code)
        + "</code></pre><p class='small'>"
        + escape(_tr(lang, 'Direct paths support `.npy`, `.npz`, `.csv`, `.txt`, and `.json`.', '直接路径支持 `.npy`、`.npz`、`.csv`、`.txt` 和 `.json`。'))
        + "</p>",
    )
    body += _card(
        _tr(lang, '2) Start from pandas or arrays', '2）从 pandas 或数组开始'),
        "<pre><code>"
        + escape(dataframe_code)
        + "</code></pre><p class='small'>"
        + escape(_tr(lang, 'Use this path when you need explicit time values or channel names.', '当你需要显式指定时间列或通道名时，走这条路径。'))
        + "</p>",
    )
    body += _card(
        _tr(lang, '3) CSV / TXT rule', '3）CSV / TXT 规则'),
        _bullets([
            _tr(lang, 'The file should be numeric.', '文件内容应当是纯数值。'),
            _tr(lang, 'If the first column is monotonic increasing, it is treated as time.', '如果第一列单调递增，它会被当作时间轴。'),
            _tr(lang, 'Otherwise the loaded matrix follows the normal shape rules below.', '否则加载出来的矩阵会按下面的 shape 规则解释。'),
        ]),
    )
    body += _card(
        _tr(lang, '4) Want one human-first example?', '4) Want one human-first example?'),
        "<p>"
        + escape(_tr(lang, 'Open the CSV example if you want the sklearn-style path: read a file, call `report(...)`, then open `report.html`.', 'Open the CSV example if you want the sklearn-style path: read a file, call `report(...)`, then open `report.html`.'))
        + "</p><p><a href='examples/csv_to_report.html'><code>csv_to_report</code></a> · <a href='examples/npy_to_handoff_bundle.html'><code>npy_to_handoff_bundle</code></a></p>",
    )
    body += "</div>"
    body += f"<div class='section'><h3>{escape(_tr(lang, 'Shape rules that decide what TSDataForge sees', '决定 TSDataForge 如何理解数据的 shape 规则'))}</h3>"
    body += _table(
        [_tr(lang, 'Input shape', '输入 shape'), _tr(lang, 'Interpretation', '解释方式')],
        rows,
    )
    body += "</div></div>"
    return body


def _route_title(route_id: str, title: str, summary: str, lang: str) -> tuple[str, str]:
    if lang.startswith("zh") and route_id in ROUTE_ZH:
        return ROUTE_ZH[route_id]
    return title, summary



def _route_table(lang: str, *, base: str = "") -> str:
    headers = [
        _tr(lang, "EDA finding", "EDA 发现"),
        _tr(lang, "Interpretation", "解释"),
        _tr(lang, "Recommended tasks", "推荐任务"),
        _tr(lang, "Examples", "案例"),
        _tr(lang, "API entry points", "API 入口"),
        _tr(lang, "FAQ", "FAQ"),
    ]
    rows: list[list[str]] = []
    for route in common_eda_finding_routes():
        title, summary = _route_title(route.route_id, route.title, route.summary, lang)
        examples = '<br/>'.join([f"<a href='{base}examples/{escape(ex_id)}.html'><code>{escape(ex_id)}</code></a>" for ex_id in route.example_ids]) or '—'
        apis = '<br/>'.join([f"<code>{escape(name)}</code>" for name in route.api_names]) or '—'
        faqs = '<br/>'.join([f"<a href='{base}faq.html#{escape(fid)}'><code>{escape(fid)}</code></a>" for fid in route.faq_ids]) or '—'
        tasks = '<br/>'.join([f"<code>{escape(task)}</code>" for task in route.recommended_tasks]) or '—'
        rows.append([escape(title), escape(summary), tasks, examples, apis, faqs])
    return _table(headers, rows)



def _positioning_table(matrix: PositioningMatrix, *, lang: str) -> str:
    headers = [
        _tr(lang, "Library", "库"),
        _tr(lang, "Best at", "最擅长什么"),
        _tr(lang, "How TSDataForge differs", "TSDataForge 的差异"),
        _tr(lang, "Use together when", "适合一起用的时候"),
    ]
    rows: list[list[str]] = []
    for item in matrix.profiles:
        best = '; '.join(item.best_for[:3]) or '—'
        diff = item.tsdataforge_difference if not lang.startswith("zh") else (item.tsdataforge_difference_zh or item.tsdataforge_difference)
        combo = item.combine_pattern if not lang.startswith("zh") else (item.combine_pattern_zh or item.combine_pattern)
        title = f"<a href='{escape(item.official_url)}'><strong>{escape(item.title)}</strong></a>"
        rows.append([title, escape(best), escape(diff), escape(combo)])
    return _table(headers, rows)


def _companion_cards(lang: str) -> str:
    queries = [
        (
            _tr(lang, "I need feature extraction for downstream ML", "我需要面向下游机器学习的特征提取"),
            "feature extraction classification regression descriptors",
        ),
        (
            _tr(lang, "I need forecasting models and backtesting", "我需要 forecasting 模型和回测"),
            "forecasting backtesting probabilistic model zoo anomaly detection",
        ),
        (
            _tr(lang, "I need shape-based similarity or clustering", "我需要 shape-based similarity 或 clustering"),
            "dtw clustering similarity motifs segmentation matrix profile",
        ),
        (
            _tr(lang, "I need generic profiling around the same project", "同一个项目里我还需要通用 profiling"),
            "profiling report metadata tabular",
        ),
    ]
    cards = []
    for title, query in queries:
        items = recommend_companions(query, top_k=2, language=lang)
        bullets = []
        for item in items:
            one = item.one_liner if not lang.startswith("zh") else (item.one_liner_zh or item.one_liner)
            combo = item.combine_pattern if not lang.startswith("zh") else (item.combine_pattern_zh or item.combine_pattern)
            bullets.append(f"<strong>{escape(item.title)}</strong>: {escape(one)}<br/><span class='small'>{escape(combo)}</span>")
        cards.append(_card(title, '<ul>' + ''.join(f'<li>{b}</li>' for b in bullets) + '</ul>'))
    return "<div class='grid'>" + ''.join(cards) + "</div>"


def _positioning_page(lang: str, matrix: PositioningMatrix) -> str:
    body = _hero(
        _tr(lang, "Positioning: where TSDataForge fits, what it does not replace, and what it combines with", "生态定位：TSDataForge 适合放在哪里、不替代什么、又适合和谁一起用"),
        _tr(lang, "TSDataForge is not a winner-takes-all time-series library. It is a deliberate layer around sequence assets: structure, explanation, taskification, and low-token handoff. This page helps users decide when TSDataForge is the main library, when it is the surrounding layer, and which companion package should take over next.", "TSDataForge 不是那种赢家通吃的时间序列库。它是一层围绕序列资产展开的能力：structure、explanation、taskification 和低 token handoff。这个页面帮助用户判断：什么时候 TSDataForge 是主库，什么时候它是外围层，接下来又该由哪个 companion package 接手。"),
        ["ecosystem fit", "companion stack", "not another model zoo", "agent-friendly"],
        pills=[f"{len(matrix.profiles)} libraries", "README-ready", "docs-ready"],
    )
    body += _toc([
        ("thesis", _tr(lang, "Core thesis", "核心定位")),
        ("table", _tr(lang, "Comparison matrix", "对比矩阵")),
        ("combine", _tr(lang, "Who to combine with", "适合一起用的库")),
        ("agent", _tr(lang, "Why this matters for agents and GitHub", "为什么这对 agent 和 GitHub 很重要")),
    ])
    thesis = matrix.thesis if not lang.startswith("zh") else matrix.thesis_zh
    body += f"<div class='section' id='thesis'><div class='grid'>"
    body += _card(_tr(lang, 'The one-sentence position', '一句话定位'), f"<p>{escape(thesis)}</p>")
    body += _card(_tr(lang, 'What TSDataForge does not try to own', 'TSDataForge 不试图独占的部分'), _bullets([
        _tr(lang, 'the whole forecasting model zoo', '整个 forecasting model zoo'),
        _tr(lang, 'the whole estimator ecosystem', '整个 estimator 生态'),
        _tr(lang, 'all feature extraction workflows', '所有 feature extraction 工作流'),
        _tr(lang, 'every simulator or domain-specific data source', '所有仿真器或领域数据源'),
    ]))
    body += _card(_tr(lang, 'What it tries to make much better', '它真正想显著做好的部分'), _bullets([
        _tr(lang, 'real-data understanding before model choice', '在选模型前先理解真实数据'),
        _tr(lang, 'one base dataset feeding many task datasets', '一份基础数据派生多个 task dataset'),
        _tr(lang, 'self-explaining assets for teams, docs, and agents', '面向团队、文档和 agent 的自解释资产'),
        _tr(lang, 'public surfaces that teach the next user', '能教会下一个用户的公共表层'),
    ]))
    body += "</div></div>"
    body += f"<div class='section' id='table'><div class='kicker'>Comparison</div><h2>{escape(_tr(lang, 'How TSDataForge differs from adjacent libraries', 'TSDataForge 与邻近库的差异'))}</h2><p class='muted'>{escape(_tr(lang, 'The goal here is not to claim that one library should replace all the others. The useful question is where each library sits in the workflow and which handoff between them is the cleanest.', '这里不是要宣称一个库能替代所有其他库。更有用的问题是：每个库位于工作流的什么位置，以及它们之间哪种交接最干净。'))}</p>{_positioning_table(matrix, lang=lang)}</div>"
    body += f"<div class='section' id='combine'><div class='kicker'>Companion stack</div><h2>{escape(_tr(lang, 'If TSDataForge is your asset layer, who should you pair it with next?', '如果 TSDataForge 是你的资产层，接下来适合搭配谁？'))}</h2>{_companion_cards(lang)}</div>"
    body += f"<div class='section' id='agent'><div class='grid'>"
    body += _card(_tr(lang, 'Why this matters for agent workflows', '为什么这对 agent workflow 很重要'), _bullets([
        _tr(lang, 'Agents should not infer the data contract from raw arrays and scattered notebook code.', 'agent 不应该从原始数组和零散 notebook 代码里反推数据契约。'),
        _tr(lang, 'A positioning page lowers drift by making the package boundary explicit.', '清晰的定位页能通过明确包边界来降低漂移。'),
        _tr(lang, 'Companion-library guidance prevents the package from sounding like an unrealistic all-in-one claim.', '明确的 companion-library 指南能避免让这个包看起来像不现实的 all-in-one 宣称。'),
    ]))
    body += _card(_tr(lang, 'Why this matters for a GitHub README', '为什么这对 GitHub README 很重要'), _bullets([
        _tr(lang, 'New visitors decide in seconds whether the package fits their workflow.', '新访客往往在几秒内决定这个包是否适合自己的 workflow。'),
        _tr(lang, 'A good README answers what the package is, what it is not, and who should use it next to existing tools.', '好的 README 会同时回答：它是什么、它不是什么、以及在已有工具旁边谁该用它。'),
        _tr(lang, 'Positioning text reduces support load because fewer users arrive with the wrong expectations.', '定位文本能降低支持负担，因为带着错误预期来的用户会更少。'),
    ]))
    body += "</div></div>"
    return _page('TSDataForge Positioning', body, lang=lang, slug='positioning.html')


def _example_cards(examples: list[ExampleRecipe], *, lang: str, prefix: str = "examples/") -> str:
    cards = []
    for ex in examples:
        category = _tr(lang, *CATEGORY_LABELS.get(ex.category, (ex.category, ex.category)))
        cards.append(
            _card(
                ex.title,
                f"<p>{escape(ex.summary)}</p>"
                f"<p><strong>{escape(_tr(lang, 'Goal', '目标'))}:</strong> {escape(ex.goal)}</p>"
                f"<p class='small'>{escape(category)} · {escape(ex.difficulty)} · {escape(', '.join(ex.outputs) or 'none')}</p>"
                f"<p><a href='{prefix}{escape(ex.example_id)}.html'>{escape(_tr(lang, 'Open example', '查看案例'))}</a></p>",
            )
        )
    return "<div class='grid'>" + ''.join(cards) + "</div>"



def _tutorial_cards(items: list[TutorialTrack], *, lang: str) -> str:
    cards = []
    for item in items:
        steps = item.steps if not lang.startswith("zh") else (item.steps_zh or item.steps)
        outcomes = item.outcomes if not lang.startswith("zh") else (item.outcomes_zh or item.outcomes)
        title = item.title if not lang.startswith("zh") else (item.title_zh or item.title)
        summary = item.summary if not lang.startswith("zh") else (item.summary_zh or item.summary)
        step_html = ''.join([f"<li>{escape(step)}</li>" for step in steps[:3]])
        cards.append(
            _card(
                title,
                f"<p>{escape(summary)}</p>"
                f"<p class='small'>{escape(_tr(lang, 'Estimated time', '预计时间'))}: {item.estimated_minutes} min · {escape(', '.join(item.audience[:3]))}</p>"
                f"<ul>{step_html}</ul>"
                f"<p class='small'>{escape(_tr(lang, 'Outcomes', '输出'))}: {escape(', '.join(outcomes))}</p>",
            )
        )
    return "<div class='grid'>" + ''.join(cards) + "</div>"



def _playbook_cards(items: list[Playbook], *, lang: str, starter_prefix: str = "starters/") -> str:
    cards = []
    for item in items:
        starter_href = f"starter-kits.html#{item.starter_id}"
        examples = ''.join([f"<span class='badge'><a href='examples/{escape(ex_id)}.html'><code>{escape(ex_id)}</code></a></span>" for ex_id in item.example_ids[:4]])
        apis = ''.join([f"<span class='badge'><code>{escape(api)}</code></span>" for api in item.api_names[:4]])
        cards.append(
            _card(
                item.title,
                f"<p>{escape(item.summary)}</p>"
                f"<p><strong>{escape(_tr(lang, 'Use this when', '适合在什么情况下使用'))}:</strong> {escape(item.when_to_use)}</p>"
                f"<p><strong>{escape(_tr(lang, 'Promise', '你能得到什么'))}:</strong> {escape(item.promise)}</p>"
                f"<div class='badges'>{_environment_badges(item.environment_ids, lang=lang)}</div>"
                f"<div class='pills'>{_scenario_badges(item.scenario_ids, lang=lang)}</div>"
                f"<p class='small'>{escape(_tr(lang, 'Examples', '案例'))}: {examples or '—'}</p>"
                f"<p class='small'>{escape(_tr(lang, 'APIs', 'API'))}: {apis or '—'}</p>"
                f"<p><a href='{starter_href}'>{escape(_tr(lang, 'Open the matching starter project', '打开对应起步项目'))}</a></p>",
            )
        )
    return "<div class='grid'>" + ''.join(cards) + "</div>"



def _starter_cards(items: list[StarterKit], *, lang: str) -> str:
    cards = []
    for item in items:
        notebook_links = ''.join([f"<span class='badge'><a href='notebooks/{escape(tid)}.ipynb'><code>{escape(tid)}.ipynb</code></a></span>" for tid in item.tutorial_ids])
        cards.append(
            _card(
                item.title,
                f"<p>{escape(item.summary)}</p>"
                f"<p><strong>{escape(_tr(lang, 'Best for', '最适合'))}:</strong> {escape(', '.join(item.best_for))}</p>"
                f"<div class='badges'>{_environment_badges(item.environment_ids, lang=lang)}</div>"
                f"<p class='small'>{escape(_tr(lang, 'Files generated', '会生成的文件'))}: {escape(', '.join(item.generated_files))}</p>"
                f"<p class='small'>{escape(_tr(lang, 'Included notebooks', '包含的 notebook'))}: {notebook_links or '—'}</p>"
                f"<p><a href='starters/{escape(item.starter_id)}/README.md'>{escape(_tr(lang, 'Open project README', '打开项目 README'))}</a></p>",
            )
        )
    return "<div class='grid'>" + ''.join(cards) + "</div>"


def _showcase_specs(lang: str) -> list[dict[str, str]]:
    return [
        {
            "group": "public",
            "section_id": "ecg",
            "scenario": "ecg_public",
            "title": _tr(lang, "Public ECG arrhythmia handoff", "公开 ECG 心律失常交接"),
            "summary": _tr(lang, "Real ECG windows from a public MIT-BIH excerpt bundled with SciPy. Best next step: event or anomaly review.", "真实 ECG 窗口，适合先看事件检测或异常检测。"),
            "command": "python -m tsdataforge demo --scenario ecg_public --output ecg_bundle",
        },
        {
            "group": "public",
            "section_id": "macro",
            "scenario": "macro_public",
            "title": _tr(lang, "Public US macro handoff", "公开美国宏观序列交接"),
            "summary": _tr(lang, "Inflation, unemployment, and T-bill windows from public macro data. Best next step: regime or forecasting review.", "基于公开宏观数据的通胀、失业和短期利率窗口，适合先看 regime 或 forecasting。"),
            "command": "python -m tsdataforge demo --scenario macro_public --output macro_bundle",
        },
        {
            "group": "public",
            "section_id": "climate",
            "scenario": "climate_public",
            "title": _tr(lang, "Public climate CO2 handoff", "公开气候 CO2 序列交接"),
            "summary": _tr(lang, "Weekly atmospheric CO2 observations with trend, seasonality, and missingness.", "真实周度大气 CO2 观测，带趋势、周期和缺失。"),
            "command": "python -m tsdataforge demo --scenario climate_public --output climate_bundle",
        },
        {
            "group": "public",
            "section_id": "sunspots",
            "scenario": "sunspots_public",
            "title": _tr(lang, "Public sunspot cycle handoff", "公开太阳黑子周期交接"),
            "summary": _tr(lang, "Physical-science sequence with long cycles and clean periodic structure.", "面向物理科学的长周期序列，适合先看周期结构和 report。"),
            "command": "python -m tsdataforge demo --scenario sunspots_public --output sunspots_bundle",
        },
        {
            "group": "synthetic",
            "section_id": "icu",
            "scenario": "icu_vitals",
            "title": _tr(lang, "Synthetic ICU vitals handoff", "合成 ICU 生命体征交接"),
            "summary": _tr(lang, "Deterministic critical-care style bundle for demos and testing.", "适合演示和测试的 ICU 风格合成 bundle。"),
            "command": "python -m tsdataforge demo --scenario icu_vitals --output icu_bundle",
        },
        {
            "group": "synthetic",
            "section_id": "macro_regime",
            "scenario": "macro_regime",
            "title": _tr(lang, "Synthetic macro regime handoff", "合成宏观 regime 交接"),
            "summary": _tr(lang, "Deterministic regime-switching macro demo.", "适合稳定复现的宏观 regime 切换 demo。"),
            "command": "python -m tsdataforge demo --scenario macro_regime --output macro_regime_bundle",
        },
        {
            "group": "synthetic",
            "section_id": "factory",
            "scenario": "factory_sensor",
            "title": _tr(lang, "Synthetic factory sensor handoff", "合成工厂传感器交接"),
            "summary": _tr(lang, "Maintenance and drift flavored sensor bundle for industrial demos.", "适合工业演示的工厂传感器 drift / maintenance bundle。"),
            "command": "python -m tsdataforge demo --scenario factory_sensor --output factory_bundle",
        },
    ]


def _showcase_bundle_href(lang: str, scenario: str, artifact: str) -> str:
    prefix = "../" if lang.startswith("zh") else ""
    return f"{prefix}showcase-bundles/{scenario}/{artifact}"


def _showcase_link_row(lang: str, scenario: str) -> str:
    links = [
        (_tr(lang, "Open full report", "查看完整报告"), "report.html"),
        (_tr(lang, "Open dataset card", "查看数据卡"), "dataset_card.md"),
        (_tr(lang, "Open handoff index", "查看 handoff 索引"), "handoff_index_min.json"),
    ]
    return "<div class='badges'>" + "".join(
        f"<a class='badge' href='{_showcase_bundle_href(lang, scenario, artifact)}'>{escape(label)}</a>"
        for label, artifact in links
    ) + "</div>"


def _showcase_card(spec: dict[str, str], *, lang: str, include_showcase_link: bool = False) -> str:
    extra = ""
    if include_showcase_link:
        extra = (
            "<p><a href='showcase.html#"
            + escape(spec["section_id"])
            + "'>"
            + escape(_tr(lang, "Open showcase page", "打开案例页"))
            + "</a></p>"
        )
    return _card(
        spec["title"],
        f"<p>{escape(spec['summary'])}</p>"
        f"<pre><code>{escape(spec['command'])}</code></pre>"
        f"{_showcase_link_row(lang, spec['scenario'])}"
        f"{extra}",
    )




def _public_surface_table(lang: str) -> str:
    from ..surface import public_surface

    surface = public_surface()
    rows = []
    for item in surface.entrypoints:
        rows.append([f"<code>{escape(item.signature)}</code>", escape(item.one_liner), f"<code>{escape(item.returns)}</code>", escape(item.why_exists)])
    return _table([_tr(lang, 'API', 'API'), _tr(lang, 'What it does', '它做什么'), _tr(lang, 'Returns', '返回什么'), _tr(lang, 'Why it exists', '为什么存在')], rows)


def _public_surface_cards(lang: str) -> str:
    from ..surface import public_surface

    surface = public_surface()
    cards = []
    for item in surface.entrypoints:
        cards.append(_card(item.name, f"<p><code>{escape(item.signature)}</code></p><p>{escape(item.one_liner)}</p><p class='small'><strong>{escape(_tr(lang, 'Returns', '返回'))}:</strong> <code>{escape(item.returns)}</code></p><p class='small'>{escape(item.why_exists)}</p>"))
    return "<div class='grid'>" + ''.join(cards) + "</div>"

def _environment_lookup(lang: str) -> dict[str, EnvironmentProfile]:
    return {item.env_id: item for item in environment_catalog(language=lang)}



def _scenario_lookup(lang: str) -> dict[str, ScenarioProfile]:
    return {item.scenario_id: item for item in scenario_catalog(language=lang)}



def _env_title(env_id: str, lang: str) -> str:
    item = _environment_lookup(lang).get(env_id)
    return item.title if item else env_id



def _scenario_title(scenario_id: str, lang: str) -> str:
    item = _scenario_lookup(lang).get(scenario_id)
    return item.title if item else scenario_id



def _environment_badges(env_ids: Iterable[str], *, lang: str) -> str:
    badges = ''.join([f"<span class='badge'>{escape(_env_title(env_id, lang))}</span>" for env_id in env_ids])
    return badges or f"<span class='badge'>{escape(_tr(lang, 'General Python environment', '通用 Python 环境'))}</span>"



def _scenario_badges(scenario_ids: Iterable[str], *, lang: str) -> str:
    badges = ''.join([f"<span class='pill'>{escape(_scenario_title(sid, lang))}</span>" for sid in scenario_ids])
    return badges or f"<span class='pill'>{escape(_tr(lang, 'General use', '通用用途'))}</span>"



def _scenario_cards(items: list[ScenarioProfile], *, lang: str) -> str:
    cards = []
    for item in items:
        cards.append(
            _card(
                item.title,
                f"<p>{escape(item.one_liner)}</p>"
                f"<p><strong>{escape(_tr(lang, 'Why TSDataForge fits', '为什么 TSDataForge 适合'))}:</strong> {escape(item.why_tsdataforge)}</p>"
                f"<p><strong>{escape(_tr(lang, 'Best environments', '适合环境'))}:</strong></p><div class='badges'>{_environment_badges(item.environment_ids, lang=lang)}</div>"
                f"<p class='small'>{escape(_tr(lang, 'Primary APIs', '主 API'))}: {escape(', '.join(item.api_names[:5]))}</p>"
                f"<p class='small'>{escape(_tr(lang, 'Outputs', '输出资产'))}: {escape(', '.join(item.outputs))}</p>",
            )
        )
    return "<div class='grid'>" + ''.join(cards) + "</div>"



def _environment_cards(items: list[EnvironmentProfile], *, lang: str) -> str:
    cards = []
    for item in items:
        cards.append(
            _card(
                item.title,
                f"<p>{escape(item.summary)}</p>"
                f"<p><strong>{escape(_tr(lang, 'Best for', '最适合'))}:</strong> {escape(', '.join(item.best_for))}</p>"
                f"<p><strong>{escape(_tr(lang, 'Strengths', '优势'))}:</strong> {escape('; '.join(item.strengths[:2]))}</p>"
                f"<p class='small'>{escape(_tr(lang, 'Typical outputs', '典型输出'))}: {escape(', '.join(item.typical_outputs))}</p>",
            )
        )
    return "<div class='grid'>" + ''.join(cards) + "</div>"



def _scenario_matrix(lang: str) -> str:
    headers = [
        _tr(lang, 'Scenario', '场景'),
        _tr(lang, 'What you are trying to do', '你要完成什么'),
        _tr(lang, 'Best environments', '适合环境'),
        _tr(lang, 'Primary APIs', '主 API'),
        _tr(lang, 'Start with examples', '建议先看案例'),
    ]
    rows: list[list[str]] = []
    for item in scenario_catalog(language=lang):
        envs = '<br/>'.join([f"<code>{escape(_env_title(env_id, lang))}</code>" for env_id in item.environment_ids]) or '—'
        apis = '<br/>'.join([f"<code>{escape(name)}</code>" for name in item.api_names[:5]]) or '—'
        examples = '<br/>'.join([f"<code>{escape(ex_id)}</code>" for ex_id in item.example_ids[:4]]) or '—'
        rows.append([escape(item.title), escape(item.one_liner), envs, apis, examples])
    return _table(headers, rows)



def _api_symbol_why(sym: APISymbol, lang: str) -> str:
    if lang.startswith('zh'):
        return sym.why_exists_zh or sym.why_exists or '这个接口存在，是为了让该工作流更稳定、更可解释。'
    return sym.why_exists or 'This symbol exists to keep the workflow explicit and reusable.'



def _api_symbol_summary(sym: APISymbol, lang: str) -> str:
    if lang.startswith('zh'):
        return sym.summary_zh or sym.summary or ''
    return sym.summary or TOP_API_SUMMARIES_EN.get(sym.name, 'See signature, related examples, and adjacent APIs for the fastest entry point.')



def _api_symbol_when(sym: APISymbol, lang: str) -> str:
    if lang.startswith('zh'):
        return sym.when_to_use_zh or sym.when_to_use or '按需使用。'
    return sym.when_to_use or ('Use this when the surrounding examples or route map point you here.' if sym.name not in TOP_API_SUMMARIES_EN else 'Use this when you want the shortest public entry point for this workflow.')


def _api_category_title(cat: APICategory, lang: str) -> str:
    return _tr(lang, *API_CATEGORY_LABELS.get(cat.category_id, (cat.title, cat.title)))



def _api_category_summary(cat: APICategory, lang: str) -> str:
    return _tr(lang, *API_CATEGORY_SUMMARIES.get(cat.category_id, (cat.summary, cat.summary)))





def _example_run_steps(ex: ExampleRecipe, lang: str) -> tuple[str, ...]:
    if ex.example_id == "csv_to_report":
        return (
            _tr(lang, "Read your CSV with pandas and pick one time column plus one or more signal columns.", "Read your CSV with pandas and pick one time column plus one or more signal columns."),
            _tr(lang, "Pass `values` and `time=` into `report(...)` to save the first human-readable `report.html`.", "Pass `values` and `time=` into `report(...)` to save the first human-readable `report.html`."),
            _tr(lang, "If you want a shareable folder, call `handoff(...)` and open `report.html` inside the saved bundle first.", "If you want a shareable folder, call `handoff(...)` and open `report.html` inside the saved bundle first."),
        )
    if ex.example_id == "npy_to_handoff_bundle":
        return (
            _tr(lang, "Point `report(...)` and `handoff(...)` at your own `.npy` file.", "Point `report(...)` and `handoff(...)` at your own `.npy` file."),
            _tr(lang, "Run the script once and let TSDataForge save the report plus the handoff folder.", "Run the script once and let TSDataForge save the report plus the handoff folder."),
            _tr(lang, "Open `report.html` first, then use the bundle sidecars only when you need the next task or a machine-readable handoff.", "Open `report.html` first, then use the bundle sidecars only when you need the next task or a machine-readable handoff."),
        )
    if "report.html" in ex.outputs:
        return (
            _tr(lang, "Copy the code into a notebook or script and replace the data source or parameters with your own.", "Copy the code into a notebook or script and replace the data source or parameters with your own."),
            _tr(lang, "Run the example once without adding extra abstractions.", "Run the example once without adding extra abstractions."),
            _tr(lang, "Open the saved report or printed output before deciding what to model next.", "Open the saved report or printed output before deciding what to model next."),
        )
    return (
        _tr(lang, "Copy the smallest code block that matches your goal.", "Copy the smallest code block that matches your goal."),
        _tr(lang, "Replace the example input or parameters with your own data or task settings.", "Replace the example input or parameters with your own data or task settings."),
        _tr(lang, "Look at the first saved artifact or printed shape/schema before expanding the workflow.", "Look at the first saved artifact or printed shape/schema before expanding the workflow."),
    )



def _example_first_result(ex: ExampleRecipe, lang: str) -> str:
    if ex.example_id == "csv_to_report":
        return _tr(
            lang,
            "You should get a standalone `report.html` plus a `my_series_handoff/` directory. Open the HTML report first because that is the shortest human check that the data was interpreted correctly.",
            "You should get a standalone `report.html` plus a `my_series_handoff/` directory. Open the HTML report first because that is the shortest human check that the data was interpreted correctly.",
        )
    if ex.example_id == "npy_to_handoff_bundle":
        return _tr(
            lang,
            "You should get `report.html` inside the saved bundle plus card/context/index sidecars. Read the report first; the JSON and Markdown files are for handoff, not for the first visual check.",
            "You should get `report.html` inside the saved bundle plus card/context/index sidecars. Read the report first; the JSON and Markdown files are for handoff, not for the first visual check.",
        )
    if "report.html" in ex.outputs:
        return _tr(
            lang,
            "A saved `report.html` is the first thing to inspect. If the report matches your expectation, only then move on to taskification, cards, or downstream modeling.",
            "A saved `report.html` is the first thing to inspect. If the report matches your expectation, only then move on to taskification, cards, or downstream modeling.",
        )
    return _tr(
        lang,
        "The first result is usually the printed shape, schema, or saved artifact listed above. Use that to confirm the workflow before composing more steps.",
        "The first result is usually the printed shape, schema, or saved artifact listed above. Use that to confirm the workflow before composing more steps.",
    )



def _example_human_note(ex: ExampleRecipe, lang: str) -> str | None:
    if ex.example_id == "csv_to_report":
        return _tr(
            lang,
            "This is the human-first example for bringing your own data into TSDataForge. Replace the file path and column names, then run it like a normal Python script.",
            "This is the human-first example for bringing your own data into TSDataForge. Replace the file path and column names, then run it like a normal Python script.",
        )
    if ex.example_id == "npy_to_handoff_bundle":
        return _tr(
            lang,
            "Use this when your dataset is already saved as a NumPy array and you want the shortest public surface, not the internal container APIs.",
            "Use this when your dataset is already saved as a NumPy array and you want the shortest public surface, not the internal container APIs.",
        )
    return None


def _example_page(ex: ExampleRecipe, *, lang: str) -> str:
    routes = example_eda_routes(ex, top_k=3)
    title = ex.title
    summary = ex.summary
    goal = ex.goal
    learnings = ex.learnings
    body = (
        f"<div class='breadcrumbs'><a href='../cookbook.html'>{escape(_tr(lang, 'Examples', '案例库'))}</a> / {escape(title)}</div>"
        + _hero(title, summary, [escape(_tr(lang, *CATEGORY_LABELS.get(ex.category, (ex.category, ex.category)))), ex.difficulty, *ex.keywords[:3]], pills=[*ex.audience[:3]])
        + "<div class='section'>" + _card(_tr(lang, 'What this example solves', '这个例子解决什么'), f"<p>{escape(goal)}</p>") + "</div>"
        + "<div class='section'><div class='grid tight'>"
        + _card(_tr(lang, 'Outputs', '输出资产'), f"<p>{escape(', '.join(ex.outputs) or 'none')}</p>")
        + _card(_tr(lang, 'Related APIs', '相关 API'), ''.join([f"<span class='badge'>{escape(item)}</span>" for item in ex.related_api]) or "<p>none</p>")
        + _card(_tr(lang, 'Typical EDA triggers', '通常在什么 EDA 发现后应该看它'), "<div class='badges'>" + ''.join([f"<span class='badge'>{escape(_route_title(r.route_id, r.title, r.summary, lang)[0])}</span>" for r in routes]) + "</div>")
        + "</div></div>"
        + "<div class='section'>" + _card(_tr(lang, 'Code', '代码'), f"<pre><code>{escape(ex.code)}</code></pre>") + "</div>"
        + "<div class='section'>" + _card(_tr(lang, 'What you learn', '你会学到什么'), '<ul>' + ''.join([f"<li>{escape(x)}</li>" for x in learnings]) + '</ul>') + "</div>"
        + "<div class='section'>" + _card(_tr(lang, 'Next step', '下一步'), f"<p>{escape(_tr(lang, 'Run this example first, then feed the output into EDA, taskify, or artifact cards.', '先跑这个例子，再把输出送进 EDA、taskify 或 artifact card。'))}</p>") + "</div>"
    )
    return _page(title, body, lang=lang, slug=f"examples/{ex.example_id}.html", base="../", level=1)



def _example_page_human(ex: ExampleRecipe, *, lang: str) -> str:
    routes = example_eda_routes(ex, top_k=3)
    title = ex.title
    summary = ex.summary
    goal = ex.goal
    learnings = ex.learnings
    human_note = _example_human_note(ex, lang)
    body = f"<div class='breadcrumbs'><a href='../cookbook.html'>{escape(_tr(lang, 'Examples', 'Examples'))}</a> / {escape(title)}</div>"
    body += _hero(
        title,
        summary,
        [escape(_tr(lang, *CATEGORY_LABELS.get(ex.category, (ex.category, ex.category)))), ex.difficulty, *ex.keywords[:3]],
        pills=[*ex.audience[:3]],
    )
    if human_note:
        body += (
            "<div class='section'><div class='notice'><strong>"
            + escape(_tr(lang, 'Why open this one first', 'Why open this one first'))
            + ":</strong> "
            + escape(human_note)
            + "</div></div>"
        )
    body += "<div class='section'><div class='grid'>"
    body += _card(_tr(lang, 'What this example solves', 'What this example solves'), f"<p>{escape(goal)}</p>")
    body += _card(_tr(lang, 'Run this example in three steps', 'Run this example in three steps'), _ordered_list(_example_run_steps(ex, lang)))
    body += _card(_tr(lang, 'What you should look at first', 'What you should look at first'), f"<p>{escape(_example_first_result(ex, lang))}</p>")
    body += "</div></div>"
    body += "<div class='section'><div class='grid tight'>"
    body += _card(_tr(lang, 'Outputs', 'Outputs'), f"<p>{escape(', '.join(ex.outputs) or 'none')}</p>")
    body += _card(_tr(lang, 'Related APIs', 'Related APIs'), ''.join([f"<span class='badge'>{escape(item)}</span>" for item in ex.related_api]) or "<p>none</p>")
    body += _card(_tr(lang, 'Typical EDA triggers', 'Typical EDA triggers'), "<div class='badges'>" + ''.join([f"<span class='badge'>{escape(_route_title(r.route_id, r.title, r.summary, lang)[0])}</span>" for r in routes]) + "</div>")
    body += "</div></div>"
    body += "<div class='section'>" + _card(_tr(lang, 'Code', 'Code'), f"<pre><code>{escape(ex.code)}</code></pre>") + "</div>"
    body += "<div class='section'>" + _card(_tr(lang, 'What you learn', 'What you learn'), _bullets(learnings)) + "</div>"
    body += "<div class='section'>" + _card(_tr(lang, 'Next step', 'Next step'), f"<p>{escape(_tr(lang, 'Run this example first, then feed the output into EDA, taskify, or artifact cards.', 'Run this example first, then feed the output into EDA, taskify, or artifact cards.'))}</p>") + "</div>"
    return _page(title, body, lang=lang, slug=f"examples/{ex.example_id}.html", base="../", level=1)


def _api_category_page(cat: APICategory, *, lang: str) -> str:
    cards = []
    for sym in cat.symbols:
        routes = api_eda_routes(sym, top_k=3)
        eda_badges = ''.join([f"<span class='badge'>{escape(_route_title(r.route_id, r.title, r.summary, lang)[0])}</span>" for r in routes]) or f"<span class='badge'>{escape(_tr(lang, 'No route tag', '暂无 route 标记'))}</span>"
        related = ', '.join([f"`{item}`" for item in sym.related]) or '—'
        examples = ', '.join([f"`{item}`" for item in sym.example_ids]) or '—'
        summary = _api_symbol_summary(sym, lang)
        when = _api_symbol_when(sym, lang)
        why = _api_symbol_why(sym, lang)
        envs = _environment_badges(sym.works_in, lang=lang)
        scenarios = _scenario_badges(sym.scenario_ids, lang=lang)
        cards.append(
            f"<div class='card' id='{escape(sym.name)}'>"
            f"<h4><code>{escape(sym.name)}</code></h4>"
            f"<p class='small'>{escape(sym.kind)} · <code>{escape(sym.module)}</code></p>"
            f"<pre><code>{escape(sym.signature)}</code></pre>"
            f"<p><strong>{escape(_tr(lang, 'What it does', '它是做什么的'))}:</strong> {escape(summary)}</p>"
            f"<p><strong>{escape(_tr(lang, 'Why it exists', '为什么会有这个 API'))}:</strong> {escape(why)}</p>"
            f"<p><strong>{escape(_tr(lang, 'Use it when', '什么时候该用'))}:</strong> {escape(when)}</p>"
            f"<p><strong>{escape(_tr(lang, 'Works best in', '适合环境'))}:</strong></p><div class='badges'>{envs}</div>"
            f"<p><strong>{escape(_tr(lang, 'Typical scenarios', '典型场景'))}:</strong></p><div class='pills'>{scenarios}</div>"
            f"<p><strong>{escape(_tr(lang, 'Related APIs', '相关 API'))}:</strong> {related}</p>"
            f"<p><strong>{escape(_tr(lang, 'Examples', '案例'))}:</strong> {examples}</p>"
            f"<p><strong>{escape(_tr(lang, 'Triggered by EDA findings', '常由哪些 EDA 发现触发'))}:</strong></p><div class='badges'>{eda_badges}</div>"
            f"</div>"
        )
    scenario_refs = recommend_scenarios(' '.join([cat.category_id, _api_category_title(cat, 'en'), _api_category_summary(cat, 'en')]), top_k=3, language=lang)
    scenario_html = _scenario_cards(scenario_refs, lang=lang)
    body = (
        f"<div class='breadcrumbs'><a href='../api-reference.html'>{escape(_tr(lang, 'API', 'API'))}</a> / {escape(_api_category_title(cat, lang))}</div>"
        + _hero(_api_category_title(cat, lang), _api_category_summary(cat, lang), [cat.category_id, f"{len(cat.symbols)} symbols"], pills=['signature-aware', 'scenario-aware', 'EDA-linked'])
        + "<div class='section notice'><strong>" + escape(_tr(lang, 'Reading tip', '阅读提示')) + ":</strong> " + escape(_tr(lang, 'Use these category pages when you know the broad workflow but still need to choose the exact public API. Each symbol explains what it does, why it exists, which environments it fits, and which scenarios usually trigger it.', '当你已经知道大致工作流，但还需要选择具体公共 API 时，就看这些分类页。每个符号都会解释它做什么、为什么存在、适合什么环境，以及通常在哪些场景下触发。')) + "</div>"
        + "<div class='section'><h2>" + escape(_tr(lang, 'Typical scenarios for this category', '这个分类常见的使用场景')) + "</h2>" + scenario_html + "</div>"
        + "<div class='section'>" + ''.join(cards) + "</div>"
    )
    return _page(f"TSDataForge API / {_api_category_title(cat, lang)}", body, lang=lang, slug=f"api/{cat.category_id}.html", base="../", level=1)

def _index_page(lang: str, catalog: list[ExampleRecipe], tutorials: list[TutorialTrack], api_ref: APIReference) -> str:
    matrix = build_positioning_matrix(language=lang)
    featured_examples = recommend_examples('quickstart eda taskify agent api', top_k=6, language=lang)
    hot_examples = recommend_examples('openclaw github stars bitcoin gold oil market similarity live', top_k=4, language=lang)
    featured_tutorials = recommend_tutorials('quickstart real data agent launch', top_k=4, language=lang)
    featured_scenarios = recommend_scenarios('real data benchmark control causal docs adoption', top_k=6, language=lang)
    featured_envs = recommend_environments('notebook script agent docs release', top_k=4, language=lang)
    body = _hero(
        _tr(lang, 'Turn raw time-series files into reports, handoff bundles, and agent-ready next actions', 'TSDataForge：把原始时间序列文件变成报告、handoff bundle 和 agent-ready next actions'),
        _tr(lang, 'Use one command or one function call to turn a time-series asset into a report, a dataset card, a compact context, and a clear next step.', '用一条命令或一次函数调用，把时序数据资产变成报告、数据卡、紧凑上下文和明确下一步。'),
        ['dataset reports', 'handoff bundles', 'agent-ready assets', 'real public demos', 'schema-first'],
        pills=[f'{len(catalog)} examples', f'{len(tutorials)} tutorial tracks', f'{api_ref.n_symbols} public symbols'],
    )
    body += _home_identity_strip(lang)
    body += _toc([
        ('what', _tr(lang, 'What the package is', '这个包到底是干什么的')),
        ('why', _tr(lang, 'Why it exists', '为什么会有这个包')),
        ('handoff', _tr(lang, 'The shortest happy path', '最短 happy path')),
        ('flagship', _tr(lang, 'Three real public demos', '三个真实公开数据案例')),
        ('surface', _tr(lang, 'The five APIs to remember', '最该记住的五个 API')),
        ('positioning', _tr(lang, 'How it differs from other libraries', '它与其他库的差异')),
        ('jobs', _tr(lang, 'What you can do with it', '你可以拿它做什么')),
        ('environments', _tr(lang, 'Which environment should I use', '在哪种环境下最合适')),
        ('api-map', _tr(lang, 'API map by intent', '按意图看的 API 地图')),
        ('examples', _tr(lang, 'Open these first', '先打开这些内容')),
        ('hot-now', _tr(lang, 'Public data examples', '公共数据案例')),
    ])
    body += f"<div class='section' id='flagship'><div class='kicker'>Real public demos</div><h2>{escape(_tr(lang, 'Start here: three real public demos', '先看这里：三个真实公开数据案例'))}</h2><div class='grid'>"
    for spec in [item for item in _showcase_specs(lang) if item["group"] == "public"][:3]:
        body += _showcase_card(spec, lang=lang, include_showcase_link=True)
    body += '</div></div>'
    body += f"<div class='section' id='what'><div class='grid'>"
    body += _card(_tr(lang, 'What TSDataForge is', 'TSDataForge 是什么'), '<p>' + escape(_tr(lang, 'It is a time-series asset layer: load data, explain it, package it, and route it into the next task.', '它是一个时序数据资产层：载入数据、解释数据、打包数据，并把它路由到下一步任务。')) + '</p>')
    body += _card(_tr(lang, 'What it is not', '它不是什么'), _bullets([
        _tr(lang, 'It is not a replacement for heavy simulators such as robotics or physics engines.', '它不是用来替代大型机器人或物理仿真器的。'),
        _tr(lang, 'It is not only a plotting library; visualization is the explanation layer, not the whole product.', '它也不只是画图库；可视化是解释层，而不是全部产品。'),
        _tr(lang, 'It is not only for synthetic data; wrapping and explaining real data is a first-class workflow.', '它也不只针对合成数据；包装并解释真实数据同样是一等工作流。'),
    ]))
    body += _card(_tr(lang, 'Who it is for', '它是给谁用的'), _bullets([
        _tr(lang, 'Researchers who need structure-aware benchmarks and reusable dataset assets.', '需要结构感知 benchmark 和可复用数据资产的研究人员。'),
        _tr(lang, 'Applied teams who need to understand a real dataset before choosing the task.', '需要在选任务前先理解真实数据集的应用团队。'),
        _tr(lang, 'Maintainers who want docs, examples, API manifests, and shareable report bundles.', '需要文档、案例、API manifest 和可分享报告 bundle 的维护者。'),
        _tr(lang, 'Agent or tooling builders who need compact, schema-stable interfaces.', '需要 compact 且 schema 稳定接口的 agent 或工具链构建者。'),
    ]))
    body += '</div></div>'
    body += f"<div class='section' id='why'><div class='grid'>"
    body += _card(_tr(lang, 'Why this package exists', '为什么会有这个包'), '<p>' + escape(_tr(lang, 'Time-series tooling often splits into disconnected islands: generators, model datasets, report notebooks, and ad-hoc agent prompts. TSDataForge exists to reconnect those layers so one dataset can be explained, taskified, saved, and reused without losing structure semantics.', '时间序列工具往往分裂成互不相连的岛：生成器、模型数据集、报告 notebook 以及临时 agent prompt。TSDataForge 的存在就是为了把这些层重新接起来，让同一份数据在被解释、任务化、保存和复用的过程中，不丢失结构语义。')) + '</p>')
    body += _card(_tr(lang, 'The design bet', '核心设计判断'), _bullets([
        _tr(lang, 'Real data should be described before it is modeled.', '真实数据应该先被描述，再被建模。'),
        _tr(lang, 'One base dataset should power many tasks.', '一份基础数据集应该服务多种任务。'),
        _tr(lang, 'Saved assets should carry schema, context, cards, and human-readable reports.', '保存下来的资产应该自带 schema、context、card 和可读报告。'),
        _tr(lang, 'Public APIs should explain why they exist, not only list signatures.', '公共 API 不该只列签名，也应该解释自己为什么存在。'),
    ]))
    body += _card(_tr(lang, 'The simplest way to think about it', '最简单的理解方式'), _bullets([
        'SeriesSpec → GeneratedSeries',
        'SeriesDataset → TaskDataset',
        'describe / EDA → choose task',
        'save asset → share with humans and agents',
    ]))
    body += '</div></div>'
    body += f"<div class='section' id='handoff'><div class='kicker'>Happy path</div><h2>{escape(_tr(lang, 'Start with one handoff bundle, not a long module tour', '先生成一个 handoff bundle，而不是先看很长的模块列表'))}</h2><div class='two-col'>"
    body += _card(_tr(lang, 'Recommended first call', '推荐第一条调用'), "<pre><code>from tsdataforge import handoff\n\nbundle = handoff(\n    dataset,\n    output_dir='dataset_handoff_bundle',\n    include_schemas=True,\n)\nprint(bundle.output_dir)</code></pre><p class='small'>" + escape(_tr(lang, 'This creates report + context + card + handoff index + manifest + schemas in one predictable directory.', '这会把 report + context + card + handoff index + manifest + schemas 放进一个可预测目录。')) + "</p>")
    body += _card(_tr(lang, 'Open in this order', '推荐打开顺序'), _bullets([_tr(lang, '1. report.html', '1. report.html'), _tr(lang, '2. dataset_card.md', '2. dataset_card.md'), _tr(lang, '3. dataset_context.json', '3. dataset_context.json'), _tr(lang, '4. handoff_index_min.json', '4. handoff_index_min.json'), _tr(lang, '5. choose one next action from the bundle', '5. 从 bundle 里挑一个 next action')]))
    body += "</div></div>"
    body += f"<div class='section' id='surface'><div class='kicker'>Public surface</div><h2>{escape(_tr(lang, 'The five APIs to remember first', '最先记住的五个 API'))}</h2><p class='muted'>{escape(_tr(lang, 'Everything else in TSDataForge can stay advanced for a while. These five entry points are the public product surface the README, docs, and agents should agree on.', 'TSDataForge 里的其他东西暂时都可以留在 advanced 区。这五个入口就是 README、docs 和 agent 应该共同遵守的公共产品表层。'))}</p>{_public_surface_table(lang)}</div>"
    top_companions = [item for item in matrix.profiles if item.kind != 'self'][:4]
    comp_rows = []
    for item in top_companions:
        diff = item.tsdataforge_difference if not lang.startswith('zh') else (item.tsdataforge_difference_zh or item.tsdataforge_difference)
        comp_rows.append([f"<a href='{escape(item.official_url)}'><strong>{escape(item.title)}</strong></a>", escape('; '.join(item.best_for[:2]) or '—'), escape(diff)])
    body += f"<div class='section' id='positioning'><div class='kicker'>Positioning</div><h2>{escape(_tr(lang, 'Why this is not just another time-series package', '为什么这不是又一个普通时间序列包'))}</h2><p class='muted'>{escape(_tr(lang, 'The most important adoption question is not how many APIs exist. It is whether people can tell, in one screen, when TSDataForge is the right layer and when another library should take over next.', '最重要的采用问题不是 API 有多少，而是用户能不能在一个屏幕内看懂：什么时候 TSDataForge 是对的那一层，接下来又该由哪个库接手。'))}</p>{_table([_tr(lang, 'Library', '库'), _tr(lang, 'Usually strongest at', '通常最擅长'), _tr(lang, 'TSDataForge difference', 'TSDataForge 的差异')], comp_rows)}<p><a href='positioning.html'>{escape(_tr(lang, 'Open the full ecosystem-fit page', '打开完整生态定位页'))}</a></p></div>"
    body += f"<div class='section' id='jobs'><div class='kicker'>Capabilities</div><h2>{escape(_tr(lang, 'What you can do with the package', '你可以用这个包做什么'))}</h2>{_scenario_cards(featured_scenarios, lang=lang)}</div>"
    body += f"<div class='section' id='environments'><div class='kicker'>Environments</div><h2>{escape(_tr(lang, 'Where the package works best', '这个包在哪些环境里最能发挥作用'))}</h2><p class='muted'>{escape(_tr(lang, 'Different environments change the best entry point. Start in a notebook for exploration, move to scripts for reproducible assets, use CI for docs and release checks, and use compact contexts when an agent is in the loop.', '不同环境会改变最合适的入口。探索先用 notebook，产出可复现资产用脚本，文档和发布检查用 CI，而一旦接入 agent 就优先使用 compact context。'))}</p>{_environment_cards(featured_envs, lang=lang)}</div>"
    workflow_rows = [
        [_tr(lang, 'I want one series quickly', '我想先快速生成一条序列'), '<code>generate_series</code> / <code>compose_series</code>', '<code>quickstart_univariate</code>'],
        [_tr(lang, 'I have real data and do not know the task yet', '我手里有真实数据，还不知道该做什么任务'), '<code>describe_series</code> / <code>generate_eda_report</code>', '<code>real_series_eda</code>'],
        [_tr(lang, 'I want one base dataset for many tasks', '我想让一份基础数据集服务多种任务'), '<code>generate_series_dataset</code> / <code>taskify_dataset</code>', '<code>taskify_forecasting</code>'],
        [_tr(lang, 'I want the shortest path to training arrays', '我只想最快拿到训练数组'), '<code>generate_dataset</code>', '<code>system_identification</code>'],
        [_tr(lang, 'I need an agent-friendly asset', '我需要 agent-friendly 资产'), '<code>build_agent_context</code> / <code>build_task_context</code> / <code>build_api_reference</code>', '<code>agent_context_pack</code>'],
        [_tr(lang, 'I need publishable docs and examples', '我需要可发布的文档和案例'), '<code>generate_docs_site</code> / <code>build_api_reference</code>', '<code>docs_site_generation</code>'],
        [_tr(lang, 'I want public-data case studies', '我想做公共数据案例'), '<code>fetch_github_stars_series</code> / <code>fetch_coingecko_market_chart</code> / <code>pairwise_similarity</code>', '<code>openclaw_stars_similarity</code>'],
    ]
    body += f"<div class='section' id='api-map'><div class='kicker'>API map</div><h2>{escape(_tr(lang, 'Start from intent, not from modules', '从意图进入，而不是从模块猜'))}</h2>{_table([_tr(lang, 'What you want to do', '你要做什么'), _tr(lang, 'Start with APIs', '先看哪些 API'), _tr(lang, 'Start with example', '先看哪个案例')], workflow_rows)}</div>"
    body += f"<div class='section' id='examples'><div class='grid'>" + _card(_tr(lang, 'Open these examples first', '建议先打开的案例'), _example_cards(featured_examples, lang=lang)) + _card(_tr(lang, 'Follow these tutorial tracks', '建议先走的教程路径'), _tutorial_cards(featured_tutorials, lang=lang)) + "</div></div>"
    body += f"<div class='section' id='hot-now'><div class='kicker'>Public data</div><h2>{escape(_tr(lang, 'Hot right now: public data examples that are easy to share', '当下可直接分享的公共数据案例'))}</h2><p class='muted'>{escape(_tr(lang, 'These examples use public signals such as GitHub attention, crypto, gold, and oil. They work as both tutorials and demos.', '这些案例使用 GitHub 热度、加密货币、黄金和原油等公共信号，既能教学，也能演示。'))}</p>{_example_cards(hot_examples, lang=lang)}</div>"
    featured_playbooks = recommend_playbooks('first success real data benchmark control agent', top_k=3, language=lang)
    featured_starters = recommend_starters('new user real data benchmark control agent', top_k=3, language=lang)
    body += f"<div class='section'><div class='grid'>" + _card(_tr(lang, 'Choose a workflow if you care more about the goal than the modules', '如果你更关心目标而不是模块名，就先选一条工作流'), _playbook_cards(featured_playbooks, lang=lang)) + _card(_tr(lang, 'Open a starter project if you want a ready project layout', '如果你想直接拿到项目骨架，就先打开起步项目'), _starter_cards(featured_starters, lang=lang)) + "</div></div>"
    body += f"<div class='section'><div class='notice'><strong>{escape(_tr(lang, 'If you only remember one thing', '如果你只记住一件事'))}:</strong> {escape(_tr(lang, 'TSDataForge becomes much easier once you think in this order: understand the data, create or import a reusable asset, derive the task view, then save a self-explaining artifact.', '一旦你按照这个顺序思考，TSDataForge 会简单很多：先理解数据，再创建或导入可复用资产，然后派生任务视图，最后保存自解释资产。'))}</div></div>"
    body = body.replace('Public data examples that are easy to share', 'Hot right now: public data examples that are easy to share')
    return _page('TSDataForge Docs', body, lang=lang, slug='index.html')

def _getting_started_page(lang: str) -> str:
    quick_examples = recommend_examples('quickstart dataset taskify eda', top_k=4, language=lang)
    envs = recommend_environments('notebook script real data first success', top_k=3, language=lang)
    body = _hero(
        _tr(lang, 'Quickstart: understand what the package does, then get the first meaningful success in five minutes', '快速上手：先搞清这个包做什么，再在 5 分钟内跑通第一次有意义的成功'),
        _tr(lang, 'Do not start by memorizing modules. Start by generating one report or one handoff bundle and open the saved files in order.', '不要先背模块名。先生成一份报告或一个 handoff bundle，然后按顺序打开产物。'),
        ['load_asset', 'report', 'handoff', 'taskify', 'demo'],
        pills=['copy/paste', 'first success', 'asset-first'],
    )
    body += _toc([
        ('what', _tr(lang, 'What this package is for', '这个包是干什么的')),
        ('surface', _tr(lang, 'The five APIs to learn first', '先学会的五个 API')),
        ('model', _tr(lang, 'The three objects to learn first', '先学会的三个对象')),
        ('ladder', _tr(lang, '30-second / 5-minute / 20-minute ladder', '30 秒 / 5 分钟 / 20 分钟路径')),
        ('five', _tr(lang, 'The first four code blocks', '前四段代码')),
        ('envs', _tr(lang, 'Good starting environments', '最合适的起步环境')),
        ('pitfalls', _tr(lang, 'Pitfalls', '常见坑')),
    ])
    body += f"<div class='section' id='flagship'><div class='kicker'>Real public demos</div><h2>{escape(_tr(lang, 'If you want a concrete starting point, open one of these demos', '如果你想马上看到真实效果，就先打开这几个案例'))}</h2><div class='grid'>"
    for spec in [item for item in _showcase_specs(lang) if item["group"] == "public"][:3]:
        body += _showcase_card(spec, lang=lang, include_showcase_link=True)
    body += '</div></div>'
    body += f"<div class='section' id='what'><div class='grid'>"
    body += _card(_tr(lang, 'The one-sentence answer', '一句话回答'), '<p>' + escape(_tr(lang, 'TSDataForge helps you define time-series structure, describe real data, convert sequence assets into tasks, and save everything in a form that is easier to explain, share, and automate.', 'TSDataForge 帮你定义时间序列结构、描述真实数据、把序列资产转成任务，并把这一切保存成更容易解释、分享和自动化消费的形式。')) + '</p>')
    body += _card(_tr(lang, 'If you already have real data', '如果你已经有真实数据'), '<p>' + escape(_tr(lang, 'Do not start by picking a model. Start with a dataset handoff bundle or an EDA report.', '不要先选模型，先从 describe_series 或 EDA 报告开始。')) + '</p>')
    body += _card(_tr(lang, 'If you want a benchmark', '如果你想做 benchmark'), '<p>' + escape(_tr(lang, 'Do not start from generate_dataset unless the task is fixed. Start from a reusable SeriesDataset whenever you can.', '除非任务已经固定，否则不要一上来就用 generate_dataset。能从可复用 SeriesDataset 开始就尽量从那里开始。')) + '</p>')
    body += '</div></div>'
    body += f"<div class='section' id='surface'><h2>{escape(_tr(lang, 'The five APIs to learn first', '最先要学会的五个 API'))}</h2>{_public_surface_cards(lang)}</div>"
    rows = [
        ['GeneratedSeries', _tr(lang, 'One sequence plus spec and trace.', '单条序列加 spec 和 trace。'), _tr(lang, 'Inspect structure, debug assumptions, or produce one example.', '查看结构、调试假设，或产出一个例子。')],
        ['SeriesDataset', _tr(lang, 'A reusable collection of time-series assets.', '一组可复用的时间序列资产。'), _tr(lang, 'Power several tasks from one common source.', '让多个任务共用一份来源。')],
        ['TaskDataset', _tr(lang, 'A task-specific X/y/masks/aux/schema protocol.', '任务专有的 X/y/masks/aux/schema 协议层。'), _tr(lang, 'Train, evaluate, save, or hand off to an agent.', '用于训练、评测、保存，或交给 agent。')],
    ]
    body += f"<div class='section' id='model'><h2>{escape(_tr(lang, 'The three objects to remember first', '最先记住的三个对象'))}</h2>{_table([_tr(lang, 'Object', '对象'), _tr(lang, 'What it is', '是什么'), _tr(lang, 'Why it matters', '为什么重要')], rows)}</div>"
    body += f"<div class='section'><div class='tip'><strong>{escape(_tr(lang, '60-second demo', '60 秒演示'))}:</strong><pre><code>pip install &quot;.[viz]&quot;\npython -m tsdataforge demo --output demo_bundle</code></pre><p>{escape(_tr(lang, 'Open `demo_bundle/report.html` first.', '先打开 `demo_bundle/report.html`。'))}</p></div></div>"
    blocks = [
        (
            _tr(lang, '1) Create one reusable base dataset', '1）生成一份可复用基础数据集'),
            '''from tsdataforge import load_asset, report

dataset = load_asset('demo.npy')
report(dataset, output_path='report.html')''',
            _tr(lang, 'If you already have a file, the first public move is simply load_asset(...) followed by report(...).', '如果你已经有文件，最先的公共路径就是 load_asset(...) 然后接 report(...).'),
        ),
        (
            _tr(lang, '2) Build the handoff bundle', '2）生成 handoff bundle'),
            '''from tsdataforge import handoff

bundle = handoff(
    'demo.npy',
    output_dir='dataset_handoff_bundle',
)
print(bundle.output_dir)''',
            _tr(lang, 'This is the shortest outcome-first path: one call yields report + context + card + index + next actions.', '这是最短的 outcome-first 路径：一个调用就能得到 report + context + card + index + next actions。'),
        ),
        (
            _tr(lang, '3) Open the report and pick the next task', '3）打开 report 并决定下一个任务'),
            '''# open dataset_handoff_bundle/report.html first
# then read dataset_card.md and handoff_index_min.json''',
            _tr(lang, 'Use the bundle to decide whether forecasting, anomaly, change point, or control/causal routing makes sense.', '借助 bundle 决定接下来走 forecasting、anomaly、change point，还是 control/causal。'),
        ),
        (
            _tr(lang, '4) Taskify only after the report', '4）看完 report 再任务化'),
            '''from tsdataforge import load_asset, taskify

base = load_asset('demo.npy')
forecast = taskify(base, task='forecasting', horizon=24)
forecast.save('saved_forecast_dataset')''',
            _tr(lang, 'Now the output is trainable, explainable, and shareable.', '现在这个输出既可训练、可解释，也可分享。'),
        ),
    ]
    body += f"<div class='section' id='five'><h2>{escape(_tr(lang, 'The first four code blocks to run', '最先应该跑通的四段代码'))}</h2><div class='grid'>"
    for title, code, desc in blocks:
        body += _card(title, f"<p>{escape(desc)}</p><pre><code>{escape(code)}</code></pre>")
    body += '</div></div>'
    body += f"<div class='section' id='envs'><h2>{escape(_tr(lang, 'Good starting environments', '最合适的起步环境'))}</h2>{_environment_cards(envs, lang=lang)}</div>"
    body += f"<div class='section'><h2>{escape(_tr(lang, 'Choose your next example by goal', '按目标选择下一条案例'))}</h2>{_example_cards(quick_examples, lang=lang)}</div>"
    body += f"<div class='section' id='pitfalls'><div class='warning'><strong>{escape(_tr(lang, 'Common pitfall', '常见坑'))}:</strong> {escape(_tr(lang, 'Do not force every dataset into forecasting on day one. For real data, the best first step is usually describe + EDA, not model selection.', '不要在第一天就把所有数据都强行做 forecasting。面对真实数据时，最好的第一步通常是 describe + EDA，而不是先选模型。'))}</div></div>"
    return _page('TSDataForge Quickstart', body, lang=lang, slug='getting-started.html')

def _handoff_page(lang: str) -> str:
    body = _hero(
        _tr(lang, "Handoff: the shortest dataset -> report -> next-action path", "交接路径：dataset -> report -> next action 的最短路径"),
        _tr(lang, "This page is the outcome-first center of TSDataForge. If you already have a dataset asset, the fastest useful move is to generate one handoff bundle that packages an HTML report, a compact context, a dataset card, and explicit next actions.", "这个页面是 TSDataForge 的 outcome-first 中心。如果你手里已经有一个 dataset asset，最快的有效动作就是生成一个 handoff bundle，把 HTML 报告、compact context、dataset card 和明确的 next actions 一起打包。"),
        ["report-first", "agent-friendly", "shareable artifacts"],
        pills=["handoff", "CLI", "README-ready"],
    )
    body += _toc([
        ("why", _tr(lang, "Why start here", "为什么应该从这里开始")),
        ("python", _tr(lang, "One Python call", "一个 Python 调用")),
        ("open-order", _tr(lang, "Human and agent open order", "人类和 agent 的打开顺序")),
        ("actions", _tr(lang, "Why the next-action plan is believable", "为什么 next-action plan 是可信的")),
        ("cli", _tr(lang, "One CLI command", "一个 CLI 命令")),
        ("artifacts", _tr(lang, "What the bundle contains", "bundle 里有什么")),
        ("when", _tr(lang, "When to use it", "什么时候用")),
    ])
    body += f"<div class='section' id='why'><div class='grid'>"
    body += _card(_tr(lang, 'What problem it solves', '它解决什么问题'), '<p>' + escape(_tr(lang, 'Most users do not need a long API tour on day one. They need one outcome: a dataset report that is already packaged for the next person or the next agent.', '大多数用户在第一天并不需要一长串 API 导览，他们需要的是一个结果：一份已经适合交给下一个人或下一个 agent 的 dataset report。')) + '</p>')
    body += _card(_tr(lang, 'What makes it different', '它和单独的 report / card 有什么不同'), '<p>' + escape(_tr(lang, 'The handoff bundle is not one more artifact. It is the deliberate packaging of report + context + card + manifest + next actions into one predictable directory.', 'handoff bundle 不是再多一个 artifact，而是把 report + context + card + manifest + next actions 有意识地打包成一个可预测目录。')) + '</p>')
    body += _card(_tr(lang, 'What to avoid', '要避免什么'), '<p>' + escape(_tr(lang, 'Avoid starting by pasting raw arrays into prompts or sending only a long README. Start with the bundle, then open the raw asset only if needed.', '避免一开始就把原始数组塞进 prompt，或者只给一份很长的 README。应该先从 bundle 开始，只有在必要时再打开原始资产。')) + '</p>')
    body += '</div></div>'
    body += f"<div class='section' id='open-order'><h2>{escape(_tr(lang, 'Human and agent open order', '人类和 agent 的打开顺序'))}</h2><div class='grid'>"
    body += _card(_tr(lang, 'Human open order', '人类打开顺序'), "<ol><li><code>report.html</code></li><li><code>dataset_card.md</code></li><li><code>dataset_context.json</code></li><li><code>handoff_index_min.json</code></li><li><code>action_plan.json</code></li></ol>")
    body += _card(_tr(lang, 'Agent open order', 'agent 打开顺序'), "<ol><li><code>handoff_index_min.json</code></li><li><code>dataset_context.json</code></li><li><code>dataset_card.md</code></li><li><code>action_plan.json</code></li><li><code>recommended_next_step</code></li></ol><p class='small'>" + escape(_tr(lang, 'Do not open `handoff_bundle.json` unless a required field is missing.', '除非缺少必要字段，否则不要先打开 `handoff_bundle.json`。')) + "</p>")
    body += '</div></div>'
    body += f"<div class='section' id='actions'><div class='notice'><strong>{escape(_tr(lang, 'What changed in the action plan', 'next-action plan 做了什么改进'))}:</strong> {escape(_tr(lang, 'The bundle now separates already-done work from the recommended next step, so the first non-open action no longer tells you to rebuild the bundle you already have.', 'bundle 现在会把 already-done 和 recommended next step 分开，因此第一条非打开动作不会再让你重建已经存在的 bundle。'))}</div></div>"
    body += f"<div class='section' id='python'><h2>{escape(_tr(lang, 'One Python call', '一个 Python 调用'))}</h2>"
    body += _card(
        _tr(lang, 'Recommended happy path', '推荐 happy path'),
        "<pre><code>from tsdataforge import generate_series_dataset, handoff\n\nbase = generate_series_dataset(\n    structures=['trend_seasonal_noise', 'regime_switch'],\n    n_series=24,\n    length=192,\n    seed=0,\n)\n\nbundle = handoff(\n    base,\n    output_dir='dataset_handoff_bundle',\n    include_schemas=True,\n)\nprint(bundle.output_dir)</code></pre>"
        + "<p class='small'>"
        + escape(_tr(lang, 'Open `report.html` first. Then read `dataset_card.md` and `dataset_context.json` before picking the next task.', '先打开 `report.html`，再读 `dataset_card.md` 和 `dataset_context.json`，最后再决定下一个任务。'))
        + "</p>"
    )
    body += '</div>'
    body += f"<div class='section' id='cli'><h2>{escape(_tr(lang, 'One CLI command', '一个 CLI 命令'))}</h2>"
    body += _card(
        _tr(lang, 'CLI for saved arrays or external assets', '面向保存数组或外部资产的 CLI'),
        "<pre><code>tsdataforge handoff my_dataset.npy --output handoff_bundle\ntsdataforge report my_dataset.npy --output report.html</code></pre>"
        + "<p class='small'>"
        + escape(_tr(lang, 'Use the CLI when you want a shortest path for teammates who should not have to write Python glue first.', '当你希望队友不必先写 Python glue 就能跑通最短路径时，用 CLI。'))
        + "</p>"
    )
    body += '</div>'
    rows = [
        ['report.html', _tr(lang, 'Outcome-first HTML EDA report', '面向结果的 HTML EDA 报告')],
        ['dataset_context.json / .md', _tr(lang, 'Low-token context for prompts and agent handoff', '给 prompt 和 agent handoff 用的低 token context')],
        ['dataset_card.json / .md', _tr(lang, 'Human + machine-readable asset summary', '可供人和机器同时读取的资产摘要')],
        ['handoff_index_min.json / .md', _tr(lang, 'Smallest agent-first entry contract', '最小的 agent-first 入口契约')],
        ['handoff_index_min.json / .md', _tr(lang, 'Smallest first-entry contract for agents and a compact human-readable route', '给人和 agent 使用的紧凑路由图')],
        ['action_plan.json / .md', _tr(lang, 'Detailed already_done / recommended / optional plan', '详细 already_done / recommended / optional 计划')],
        ['manifest / handoff_bundle', _tr(lang, 'One predictable inventory of what was saved and what to do next', '把保存内容和下一步动作集中到一个可预测清单里')],
        ['asset/', _tr(lang, 'Optional raw dataset export', '可选的原始数据集导出')],
    ]
    body += f"<div class='section' id='artifacts'><h2>{escape(_tr(lang, 'What the bundle contains', 'bundle 里有什么'))}</h2>{_table([_tr(lang, 'Artifact', '产物'), _tr(lang, 'Why it is there', '为什么在这里')], rows)}</div>"
    body += f"<div class='section' id='when'><h2>{escape(_tr(lang, 'Use it first when ...', '在这些情况下应优先使用'))}</h2>"
    body += _card(
        _tr(lang, 'Good fit', '适合场景'),
        _bullets([
            _tr(lang, 'You have a real dataset and need the first shareable explanation artifact.', '你拿到真实数据，需要第一份可分享解释资产。'),
            _tr(lang, 'You want to hand a dataset to another researcher without losing task semantics.', '你想把数据交给另一位研究者，同时不丢失任务语义。'),
            _tr(lang, 'An agent should read a compact context instead of raw arrays.', '你希望 agent 读 compact context，而不是原始数组。'),
            _tr(lang, 'You want a repeatable public demo for GitHub or docs.', '你想为 GitHub 或 docs 准备一条可重复的公开演示路径。'),
        ])
    )
    body += '</div>'
    return _page('TSDataForge Handoff', body, lang=lang, slug='handoff.html')


def _showcase_page(lang: str) -> str:
    body = _hero(
        _tr(lang, "Showcase: open the finished bundle, not just the command", "案例展示：不仅给命令，也直接给完整 bundle"),
        _tr(lang, "Each demo below links to the saved report, dataset card, and handoff index so you can inspect the full output directly.", "下面每个案例都直接链接到保存好的报告、数据卡和 handoff 索引，可以直接看完整输出。"),
        ["real-public-data", "report-first", "shareable"],
        pills=["report", "card", "handoff index"],
    )
    body += _toc([
        ("how", _tr(lang, "How to read a showcase bundle", "怎么阅读一个 showcase bundle")),
        ("public", _tr(lang, "Real public-data demos", "真实公开数据案例")),
        ("synthetic", _tr(lang, "Synthetic demos", "合成案例")),
    ])
    body += f"<div class='section' id='how'><div class='grid'>"
    body += _card(_tr(lang, 'Open files in this order', '建议打开顺序'), _bullets([
        _tr(lang, '1. report.html', '1. report.html'),
        _tr(lang, '2. dataset_card.md', '2. dataset_card.md'),
        _tr(lang, '3. handoff_index_min.json', '3. handoff_index_min.json'),
    ]))
    body += _card(_tr(lang, 'What this page proves', '这页证明什么'), _bullets([
        _tr(lang, 'The package can say something useful about real time-series data.', '这个包能对真实时序数据给出有用的解释。'),
        _tr(lang, 'The saved bundle is readable by both humans and agents.', '保存下来的 bundle 同时适合人和 agent。'),
        _tr(lang, 'The next step is explicit instead of hidden in a notebook.', '下一步是显式写出来的，不是藏在 notebook 里。'),
    ]))
    body += "</div></div>"
    body += f"<div class='section' id='public'><div class='kicker'>Real public data</div><h2>{escape(_tr(lang, 'Real public-data demos', '真实公开数据案例'))}</h2><div class='grid'>"
    for spec in [item for item in _showcase_specs(lang) if item["group"] == "public"]:
        body += f"<div id='{escape(spec['section_id'])}'>{_showcase_card(spec, lang=lang)}</div>"
    body += "</div></div>"
    body += f"<div class='section' id='synthetic'><div class='kicker'>Synthetic</div><h2>{escape(_tr(lang, 'Synthetic demos for deterministic testing and screenshots', '适合稳定测试和截图的合成案例'))}</h2><div class='grid'>"
    for spec in [item for item in _showcase_specs(lang) if item["group"] == "synthetic"]:
        body += f"<div id='{escape(spec['section_id'])}'>{_showcase_card(spec, lang=lang)}</div>"
    body += "</div></div>"
    return _page('TSDataForge Showcase', body, lang=lang, slug='showcase.html')


def _tutorials_page(lang: str) -> str:
    tutorials = tutorial_catalog(language=lang)
    body = _hero(
        _tr(lang, "Tutorials: guided paths from first result to repeatable workflow", "教程：从第一次成功走到可复用工作流"),
        _tr(lang, "Use these pages when the API list is too low-level and you want a short path for a concrete goal.", "当 API 列表对你来说太底层时，就看这里；这里给的是面向具体目标的短路径。"),
        [f"{len(tutorials)} tracks", "example-backed", "beginner to advanced"],
        pills=["first result", "real data", "repeat usage"],
    )
    body += _toc([(item.tutorial_id, item.title if not lang.startswith('zh') else (item.title_zh or item.title)) for item in tutorials])
    body += f"<div class='section'>" + _card(_tr(lang, 'How to use this page', '怎么用这页'), "<p>" + escape(_tr(lang, "Pick one track, run it end to end, and only then open adjacent API pages.", "先挑一条教程完整跑通，然后再看相邻的 API 页面。")) + "</p>") + "</div>"
    body += f"<div class='section'><div class='tip'><strong>{escape(_tr(lang, 'Prefer the shortest path?', '想走最短路径？'))}:</strong> <a href='handoff.html'>{escape(_tr(lang, 'Open Handoff', '打开 Handoff'))}</a> · <a href='starter-kits.html'>{escape(_tr(lang, 'Open Starter Projects', '打开起步项目'))}</a> · <a href='notebooks/first-five-minutes.ipynb'><code>first-five-minutes.ipynb</code></a></div></div>"
    for item in tutorials:
        title = item.title if not lang.startswith('zh') else (item.title_zh or item.title)
        summary = item.summary if not lang.startswith('zh') else (item.summary_zh or item.summary)
        steps = item.steps if not lang.startswith('zh') else (item.steps_zh or item.steps)
        outcomes = item.outcomes if not lang.startswith('zh') else (item.outcomes_zh or item.outcomes)
        examples = ''.join([f"<li><a href='examples/{escape(ex_id)}.html'><code>{escape(ex_id)}</code></a></li>" for ex_id in item.example_ids])
        body += f"<div class='section' id='{escape(item.tutorial_id)}'><div class='two-col'>"
        body += _card(title, f"<p>{escape(summary)}</p><p class='small'>{escape(_tr(lang, 'Estimated time', '预计时间'))}: {item.estimated_minutes} min · {escape(', '.join(item.audience))}</p><ol>" + ''.join([f"<li>{escape(x)}</li>" for x in steps]) + "</ol>")
        body += _card(_tr(lang, 'Backed examples and expected outcomes', '配套案例与输出'), f"<p><strong>{escape(_tr(lang, 'Outcomes', '输出'))}:</strong> {escape(', '.join(outcomes))}</p><ul>{examples}</ul>")
        body += "</div></div>"
    return _page("TSDataForge Tutorials", body, lang=lang, slug="tutorials.html")



def _playbooks_page(lang: str) -> str:
    items = playbook_catalog(language=lang)
    body = _hero(
        _tr(lang, "Workflows: choose a goal-driven path instead of guessing from modules", "工作流：按目标选路径，而不是从模块名猜"),
        _tr(lang, "Each workflow maps one user goal to the first examples, the key APIs, and the matching starter project.", "每条工作流都把一个用户目标映射到起步案例、关键 API 和对应的起步项目。"),
        [f"{len(items)} workflows", "goal-first", "tutorial-backed"],
        pills=["first success", "real data", "benchmark", "control/causal", "agent"],
    )
    body += _toc([(item.playbook_id, item.title) for item in items])
    body += f"<div class='section'>" + _card(_tr(lang, 'Why workflows matter', '为什么要有工作流页'), '<p>' + escape(_tr(lang, 'Most users ask for the next sensible route, not for a module map. This page answers that question directly.', '大多数用户想知道的是“下一条合理路径是什么”，不是模块分布图。这页就是直接回答这个问题。')) + '</p>') + '</div>'
    body += f"<div class='section'>" + _playbook_cards(items, lang=lang) + '</div>'
    rows = []
    for item in items:
        rows.append([
            escape(item.title),
            escape(item.when_to_use),
            '<br/>'.join([f"<code>{escape(_env_title(env_id, lang))}</code>" for env_id in item.environment_ids]) or '—',
            '<br/>'.join([f"<code>{escape(api)}</code>" for api in item.api_names[:4]]) or '—',
            f"<a href='starter-kits.html#{escape(item.starter_id)}'><code>{escape(item.starter_id)}</code></a>",
        ])
    body += f"<div class='section'><h2>{escape(_tr(lang, 'Workflow matrix', '工作流矩阵'))}</h2>{_table([_tr(lang, 'Workflow', '工作流'), _tr(lang, 'Use it when', '适合在什么情况下使用'), _tr(lang, 'Best environments', '适合环境'), _tr(lang, 'APIs', 'API'), _tr(lang, 'Starter', 'Starter')], rows)}</div>"
    return _page('TSDataForge Workflows', body, lang=lang, slug='playbooks.html')



def _starter_kits_page(lang: str) -> str:
    items = starter_catalog(language=lang)
    body = _hero(
        _tr(lang, "Starter Projects: ready project layouts for the common paths", "起步项目：常见路径的现成项目骨架"),
        _tr(lang, "Use a starter project when you want files and structure, not just a page of documentation.", "如果你需要的是文件结构和项目骨架，而不只是文档页，就用起步项目。"),
        [f"{len(items)} starters", "project templates", "downloadable"],
        pills=["README", "scripts", "notebooks", "manifest"],
    )
    body += _toc([(item.starter_id, item.title) for item in items])
    body += f"<div class='section'>" + _card(_tr(lang, 'What a starter project gives you', '起步项目会给你什么'), _bullets([
        _tr(lang, 'A small README that explains the goal of the project.', '一份短 README，解释这个项目要解决什么。'),
        _tr(lang, 'Runnable scripts built from tested examples.', '由已测试案例导出的可运行脚本。'),
        _tr(lang, 'Notebook versions of the main tutorial tracks.', '主要教程路径对应的 notebook。'),
        _tr(lang, 'A manifest that tells a collaborator or an agent what the project contains.', '一份 manifest，让协作者或 agent 立刻知道项目里有什么。'),
    ])) + '</div>'
    body += f"<div class='section'>" + _starter_cards(items, lang=lang) + '</div>'
    for item in items:
        body += f"<div class='section' id='{escape(item.starter_id)}'><div class='two-col'>"
        body += _card(item.title, f"<p>{escape(item.summary)}</p><p><strong>{escape(_tr(lang, 'Best for', '最适合'))}:</strong> {escape(', '.join(item.best_for))}</p><p><a href='starters/{escape(item.starter_id)}/README.md'>{escape(_tr(lang, 'Open README', '打开 README'))}</a></p><p><a href='starters/{escape(item.starter_id)}/starter_manifest.json'>{escape(_tr(lang, 'Open manifest', '打开 manifest'))}</a></p>")
        notebook_links = ''.join([f"<li><a href='notebooks/{escape(tid)}.ipynb'><code>{escape(tid)}.ipynb</code></a> · <a href='notebooks/{escape(tid)}.py'><code>{escape(tid)}.py</code></a></li>" for tid in item.tutorial_ids])
        body += _card(_tr(lang, 'Included notebook downloads', '可下载 notebook'), f"<ul>{notebook_links}</ul><p class='small'>{escape(_tr(lang, 'These notebooks are generated from tutorial tracks and backed by tested example code.', '这些 notebooks 是从 tutorial 路径自动生成的，并且以已测试的示例代码为基础。'))}</p>")
        body += "</div></div>"
    return _page('TSDataForge Starter Projects', body, lang=lang, slug='starter-kits.html')


def _cookbook_page(lang: str, grouped: dict[str, list[ExampleRecipe]]) -> str:
    body = _hero(
        _tr(lang, "Examples: find the shortest runnable path for a real goal", "案例库：按真实目标找到最短可运行路径"),
        _tr(lang, "Examples are the main distribution unit of a research package. They should be grouped by user intent, not by internal module layout.", "案例是研究型包最重要的传播单位。它们应该按用户意图组织，而不是按内部模块堆砌。"),
        ["goal-oriented", "copy/paste", "EDA-linked"],
        pills=["quickstart", "real data", "control/causal", "agent", "launch"],
    )
    body += _toc([(k, _tr(lang, *CATEGORY_LABELS.get(k, (k, k)))) for k in grouped])
    body += f"<div class='section'>" + _card(_tr(lang, 'How to use this page', '如何使用这页'), _bullets([
        _tr(lang, "Start with the category that matches your intent.", "先找与你意图最接近的分类。"),
        _tr(lang, "Choose the closest runnable example before you read more theory.", "先挑最近的 runnable example，再读更多理论。"),
        _tr(lang, "After the example runs, add EDA, taskify, and cards/context.", "案例跑通后，再叠加 EDA、taskify 和 cards/context。"),
    ])) + "</div>"
    for key, items in grouped.items():
        body += f"<div class='section' id='{escape(key)}'><div class='kicker'>{escape(key)}</div><h2>{escape(_tr(lang, *CATEGORY_LABELS.get(key, (key, key))))}</h2>{_example_cards(items, lang=lang)}</div>"
    body += f"<div class='section'><h2>{escape(_tr(lang, 'Common EDA findings -> which examples to open next', '常见 EDA 发现 -> 应该看哪些案例'))}</h2>{_route_table(lang)}</div>"
    return _page("TSDataForge Examples", body, lang=lang, slug="cookbook.html")



def _cookbook_page_human(lang: str, grouped: dict[str, list[ExampleRecipe]]) -> str:
    body = _hero(
        _tr(lang, "Examples: find the shortest runnable path for a real goal", "Examples: find the shortest runnable path for a real goal"),
        _tr(lang, "Examples are the main distribution unit of a research package. They should be grouped by user intent, not by internal module layout.", "Examples are the main distribution unit of a research package. They should be grouped by user intent, not by internal module layout."),
        ["goal-oriented", "copy/paste", "EDA-linked"],
        pills=["quickstart", "real data", "control/causal", "agent", "launch"],
    )
    body += _toc([(k, _tr(lang, *CATEGORY_LABELS.get(k, (k, k)))) for k in grouped])
    body += f"<div class='section'>" + _card(_tr(lang, 'How to use this page', 'How to use this page'), _bullets([
        _tr(lang, "If you want to run your own dataset first, open `csv_to_report` or `npy_to_handoff_bundle` before everything else.", "If you want to run your own dataset first, open `csv_to_report` or `npy_to_handoff_bundle` before everything else."),
        _tr(lang, "Then move to the category that matches your intent.", "Then move to the category that matches your intent."),
        _tr(lang, "Choose the closest runnable example before you read more theory.", "Choose the closest runnable example before you read more theory."),
        _tr(lang, "After the example runs, add EDA, taskify, and cards/context.", "After the example runs, add EDA, taskify, and cards/context."),
    ])) + "</div>"
    body += f"<div class='section'><div class='grid'>"
    body += _card(
        _tr(lang, 'Start here with your own CSV or DataFrame', 'Start here with your own CSV or DataFrame'),
        "<p>"
        + escape(_tr(lang, 'This is the most human example in the public docs: read a table, pick columns, call `report(...)`, then open `report.html`.', 'This is the most human example in the public docs: read a table, pick columns, call `report(...)`, then open `report.html`.'))
        + "</p><p><a href='examples/csv_to_report.html'><code>csv_to_report</code></a></p>",
    )
    body += _card(
        _tr(lang, 'Already have a NumPy file?', 'Already have a NumPy file?'),
        "<p>"
        + escape(_tr(lang, 'Open the `.npy` example if your dataset is already saved as an array and you want the shortest report + handoff path.', 'Open the `.npy` example if your dataset is already saved as an array and you want the shortest report + handoff path.'))
        + "</p><p><a href='examples/npy_to_handoff_bundle.html'><code>npy_to_handoff_bundle</code></a></p>",
    )
    body += "</div></div>"
    for key, items in grouped.items():
        body += f"<div class='section' id='{escape(key)}'><div class='kicker'>{escape(key)}</div><h2>{escape(_tr(lang, *CATEGORY_LABELS.get(key, (key, key))))}</h2>{_example_cards(items, lang=lang)}</div>"
    body += f"<div class='section'><h2>{escape(_tr(lang, 'Common EDA findings -> which examples to open next', 'Common EDA findings -> which examples to open next'))}</h2>{_route_table(lang)}</div>"
    return _page("TSDataForge Examples", body, lang=lang, slug="cookbook.html")


def _taskification_page(lang: str) -> str:
    rows = [
        ["forecasting", _tr(lang, "Future-segment prediction", "未来片段预测"), _tr(lang, "y is the future horizon", "y 是未来 horizon"), "taskify_forecasting"],
        ["classification", _tr(lang, "Structure classification", "结构分类"), _tr(lang, "y is a structure label", "y 是结构标签"), "classification_benchmark"],
        ["anomaly_detection", _tr(lang, "Anomaly detection", "异常检测"), _tr(lang, "Dense anomaly mask", "逐时刻异常标签"), "anomaly_detection"],
        ["change_point_detection", _tr(lang, "Change-point detection", "变化点检测"), _tr(lang, "Dense changepoint mask", "逐时刻 changepoint mask"), "change_point_detection"],
        ["masked_reconstruction", _tr(lang, "Masked reconstruction", "masked reconstruction"), _tr(lang, "X has masks, y is the reconstruction target", "X 带 mask，y 是重建目标"), "masked_reconstruction"],
        ["contrastive", _tr(lang, "Contrastive learning", "对比学习"), _tr(lang, "Positive and negative views", "正负对比视图"), "contrastive_pairs"],
        ["system_identification", _tr(lang, "System identification", "系统辨识"), _tr(lang, "X = past [u, y], y = future y", "X = past [u,y]，y = future y"), "system_identification"],
        ["causal_response", _tr(lang, "Causal response prediction", "因果响应预测"), _tr(lang, "Outcome over future horizon", "未来 outcome"), "causal_response"],
        ["counterfactual_response", _tr(lang, "Counterfactual response", "反事实响应"), _tr(lang, "Counterfactual outcome target", "反事实输出目标"), "policy_counterfactual"],
        ["policy_value_estimation", _tr(lang, "Policy value estimation", "策略价值估计"), _tr(lang, "Return / reward target", "return / reward 目标"), "policy_value_estimation"],
        ["intervention_detection", _tr(lang, "Intervention detection", "干预检测"), _tr(lang, "Dense intervention mask", "逐时刻干预标签"), "intervention_detection"],
    ]
    body = _hero(
        _tr(lang, "Taskification: turn sequence assets into trainable protocols", "任务化：把时序资产变成可训练协议"),
        _tr(lang, "The key design choice is not how many tasks exist, but whether all tasks are derived from the same reusable base dataset layer with explicit schema.", "关键设计不是支持多少任务，而是所有任务是否都从同一个可复用基础数据层派生，并带有明确 schema。"),
        ["SeriesDataset", "TaskDataset", "schema-first", "reuse"],
        pills=["forecasting", "classification", "causal", "control", "SSL"],
    )
    body += _toc([
        ("layers", _tr(lang, "Two-layer architecture", "两层抽象")),
        ("matrix", _tr(lang, "Task matrix", "任务矩阵")),
        ("rules", _tr(lang, "Generate now or taskify later?", "直接生成还是稍后 taskify？")),
        ("routing", _tr(lang, "EDA finding -> task routing", "EDA finding -> task 路由")),
    ])
    body += f"<div class='section' id='layers'><div class='grid'>"
    body += _card("SeriesDataset", f"<p>{escape(_tr(lang, 'Raw time-series assets plus meta and optional trace.', '原始时序资产，加上 meta 和可选 trace。'))}</p>")
    body += _card("TaskDataset", f"<p>{escape(_tr(lang, 'Explicit X / y / masks / aux / schema for training, evaluation, and agent consumption.', '显式的 X / y / masks / aux / schema，供训练、评测和 agent 使用。'))}</p>")
    body += _card(_tr(lang, "Why the split matters", "为什么一定要分层"), _bullets([
        _tr(lang, "One base dataset can serve many tasks.", "一份基础数据集可以服务多个任务。"),
        _tr(lang, "Real and synthetic data can flow through the same task layer.", "真实数据和合成数据可以共用同一任务层。"),
        _tr(lang, "Cards, schema, and contexts become stable across workflows.", "cards、schema 和 context 在不同工作流里会更稳定。"),
    ]))
    body += "</div></div>"
    body += f"<div class='section' id='matrix'><h2>{escape(_tr(lang, 'Supported task matrix', '支持的任务矩阵'))}</h2>{_table([_tr(lang, 'Task', '任务'), _tr(lang, 'Use case', '用途'), _tr(lang, 'Typical y meaning', '典型 y 语义'), _tr(lang, 'Example', '案例')], rows)}</div>"
    body += f"<div class='section' id='rules'><div class='grid'>"
    body += _card(_tr(lang, 'Use generate_dataset directly when…', '直接用 generate_dataset 的情况'), _bullets([
        _tr(lang, "You already know the exact task.", "你已经知道明确任务。"),
        _tr(lang, "You want the shortest path to trainable X/y.", "你只想最快拿到可训练的 X/y。"),
        _tr(lang, "You do not need to reuse the same base dataset across tasks.", "你暂时不需要一份基础数据复用到多个任务。"),
    ]))
    body += _card(_tr(lang, 'Use generate_series_dataset + taskify when…', '先 generate_series_dataset 再 taskify 的情况'), _bullets([
        _tr(lang, "One dataset should support many tasks.", "一份数据要支持多个任务。"),
        _tr(lang, "You want EDA or dataset cards before choosing a task.", "你想先做 EDA 或数据集 card，再决定任务。"),
        _tr(lang, "You want a long-lived reusable data asset.", "你希望得到一个长期可复用的数据资产。"),
    ]))
    body += "</div></div>"
    body += f"<div class='section' id='routing'><h2>{escape(_tr(lang, 'EDA findings route directly into tasks', 'EDA 发现会直接路由到任务'))}</h2>{_route_table(lang)}</div>"
    return _page("TSDataForge Taskification", body, lang=lang, slug="taskification.html")



def _agent_playbook_page(lang: str) -> str:
    body = _hero(
        _tr(lang, "Agent Playbook: lower token cost, lower drift, better grounding", "Agent Playbook：更省 token、更少跑偏、更强 grounding"),
        _tr(lang, "The goal is not to make agents remember more. The goal is to give them less context, but better context.", "目标不是让 agent 记住更多，而是给它更少但更好的上下文。"),
        ["compact context", "schema", "cards", "example routing", "API surface"],
        pills=["low-token", "low-friction", "grounded"],
    )
    rows = [
        ["1", "build_agent_context(...)", _tr(lang, "Small, stable, and action-oriented.", "小、稳、带 next actions。")],
        ["2", _tr(lang, "Dataset/task card", "dataset/task card"), _tr(lang, "Adds intended use, caveats, and quickstart without huge token cost.", "补充 intended use、caveats 和 quickstart，同时不会太费 token。")],
        ["3", "TaskDataset.schema", _tr(lang, "The single source of truth for X/y semantics.", "X/y 语义的单一事实来源。")],
        ["4", _tr(lang, "Nearest runnable example", "最近的 runnable example"), _tr(lang, "A small working example beats a long explanation.", "一段小的可运行代码胜过长篇解释。")],
        ["5", _tr(lang, "Full EDA / README", "完整 EDA / README"), _tr(lang, "Open only when the agent truly needs deeper detail.", "只有在确实需要更多细节时再打开。")],
    ]
    body += f"<div class='section'><h2>{escape(_tr(lang, 'Recommended input order for agents', '给 agent 的输入顺序'))}</h2>{_table([_tr(lang, 'Priority', '优先级'), _tr(lang, 'Give the agent', '给什么'), _tr(lang, 'Why', '为什么')], rows)}</div>"
    flows = [
        _tr(lang, "Real data -> describe / EDA -> build_agent_context -> recommend_examples -> taskify -> train", "real data -> describe / EDA -> build_agent_context -> recommend_examples -> taskify -> train"),
        _tr(lang, "Base dataset -> dataset EDA -> taskify into 2-3 tasks -> save cards / contexts -> share", "base dataset -> dataset EDA -> taskify into 2-3 tasks -> save cards / contexts -> share"),
        _tr(lang, "Spec -> counterfactual pair -> compact comparison context -> taskify only if needed", "spec -> counterfactual pair -> compact comparison context -> taskify only if needed"),
    ]
    body += f"<div class='section'><div class='grid'>"
    body += _card(_tr(lang, 'Recommended workflows', '推荐工作流'), ''.join([f"<pre><code>{escape(flow)}</code></pre>" for flow in flows]))
    body += _card(_tr(lang, 'Anti-patterns', '反模式'), _bullets([
        _tr(lang, "Pasting raw arrays into the prompt first.", "先把原始数组粘进 prompt。"),
        _tr(lang, "Giving a TaskDataset without its schema.", "给 TaskDataset 却不附带 schema。"),
        _tr(lang, "Making the agent guess which example to open.", "让 agent 自己猜该看哪个案例。"),
        _tr(lang, "Saving base and task datasets together without distinct cards and contexts.", "基础数据集和任务数据集混着保存，不区分 card/context。"),
    ]))
    body += _card(_tr(lang, 'Why this helps package adoption', '为什么这也有助于包的传播'), _bullets([
        _tr(lang, "Agents become better onboarding layers for human users.", "agent 会成为更好的 onboarding 层。"),
        _tr(lang, "Low-token assets are easier to embed into research assistants, IDE tools, and docs copilots.", "低 token 资产更容易接入研究助手、IDE 工具和 docs copilot。"),
        _tr(lang, "Stable schema and cards reduce support load when traffic grows.", "当流量增长时，稳定 schema 和 card 能降低支持成本。"),
    ]))
    body += "</div></div>"
    return _page("TSDataForge Agent Playbook", body, lang=lang, slug="agent-playbook.html")



def _use_cases_page(lang: str) -> str:
    scenarios = scenario_catalog(language=lang)
    envs = environment_catalog(language=lang)
    body = _hero(
        _tr(lang, 'Use cases and environments: show where TSDataForge fits before people ask', '应用场景与环境：先展示 TSDataForge 适合哪里，再等别人来问'),
        _tr(lang, 'A feature list is rarely enough. Users adopt a research library faster when they can map it onto a concrete job, a concrete environment, and a concrete output they care about.', '单纯的功能列表通常不够。只有当用户能把一个研究库立刻映射到具体工作、具体环境和具体输出时，采用速度才会更快。'),
        ['scenario-first', 'environment-aware', 'task-aware'],
        pills=[f'{len(scenarios)} scenarios', f'{len(envs)} environments'],
    )
    body += _toc([
        ('scenarios', _tr(lang, 'Scenario atlas', '场景地图')),
        ('matrix', _tr(lang, 'Scenario matrix', '场景矩阵')),
        ('envs', _tr(lang, 'Environment guide', '环境指南')),
    ])
    body += f"<div class='section' id='scenarios'><div class='kicker'>Scenarios</div><h2>{escape(_tr(lang, 'What people actually use the package for', '大家实际会拿这个包做什么'))}</h2>{_scenario_cards(scenarios, lang=lang)}</div>"
    body += f"<div class='section' id='matrix'><div class='kicker'>Matrix</div><h2>{escape(_tr(lang, 'Choose by job, environment, and entry API', '按工作、环境和入口 API 来选'))}</h2><p class='muted'>{escape(_tr(lang, 'If you are unsure where to start, find the row closest to your real situation rather than the row with the most features.', '如果你不确定从哪里开始，先找和你真实情况最接近的那一行，而不是功能最多的那一行。'))}</p>{_scenario_matrix(lang)}</div>"
    body += f"<div class='section' id='envs'><div class='kicker'>Environments</div><h2>{escape(_tr(lang, 'Where each workflow works best', '每种工作流最适合在哪种环境里运行'))}</h2>{_environment_cards(envs, lang=lang)}</div>"
    body += f"<div class='section'><div class='warning'><strong>{escape(_tr(lang, 'Practical note', '实际建议'))}:</strong> {escape(_tr(lang, 'TSDataForge is strongest when it sits next to your real data or simulator, adds EDA and task semantics, and then exports assets that other people or agents can consume. It does not need to replace every upstream system to be valuable.', 'TSDataForge 最强的方式，是贴着你的真实数据或仿真器工作，补 EDA 和任务语义，然后导出别人或 agent 能消费的资产。它并不需要替代所有上游系统，才算有价值。'))}</div></div>"
    return _page('TSDataForge Use Cases', body, lang=lang, slug='use-cases.html')

def _rollout_page(lang: str) -> str:
    body = _hero(
        _tr(lang, "Release: what the public surface still needs", "发布：公开面还缺什么"),
        _tr(lang, "Once the feature surface is coherent, growth depends on information architecture, examples, linked reports, FAQs, and self-explaining assets more than on adding yet another generator.", "当功能表面已经合理后，增长更依赖信息架构、案例、联动报告、FAQ 和自解释资产，而不是再多加一个生成器。"),
        ["launch", "docs", "support", "community", "distribution"],
        pills=["bilingual", "low-friction", "shareable"],
    )
    body += _toc([
        ("stack", _tr(lang, "The minimum public stack", "最小公开栈")),
        ("capacity", _tr(lang, "How to absorb traffic", "怎么接住流量")),
        ("loop", _tr(lang, "Feedback loop", "反馈回路")),
        ("checklist", _tr(lang, "Release checklist", "发布清单")),
    ])
    body += f"<div class='section' id='stack'><div class='grid'>"
    body += _card(_tr(lang, 'Minimum public stack', '最小公开栈'), _bullets([
        _tr(lang, "A bilingual landing page with a clear first step.", "双语 landing 页，并且明确第一步。"),
        _tr(lang, "A five-minute quickstart and several tutorial tracks.", "5 分钟 quickstart 和几条教程路径。"),
        _tr(lang, "25+ runnable examples grouped by intent.", "按目标分层的 25+ runnable examples。"),
        _tr(lang, "API reference, FAQ, and linked EDA reports.", "API reference、FAQ 和 linked EDA report。"),
        _tr(lang, "Saved artifacts that travel with cards, schema, and context.", "保存资产时自带 card、schema 和 context。"),
    ]))
    body += _card(_tr(lang, 'Why bilingual matters', '为什么双语重要'), _bullets([
        _tr(lang, "English should be the default public surface for discovery and search.", "英文应该是默认的公开发现语言和搜索入口。"),
        _tr(lang, "A Chinese switch helps local teaching, onboarding, and collaboration.", "中文切换能帮助本地教学、onboarding 和协作。"),
        _tr(lang, "The key is one surface, not two disconnected documentation universes.", "关键是一套统一表层，而不是两个彼此割裂的文档宇宙。"),
    ]))
    body += "</div></div>"
    body += f"<div class='section' id='capacity'><div class='grid'>"
    capacity = [
        _tr(lang, "Every new release should ship at least one beginner tutorial and one high-signal advanced example.", "每个版本至少新增一条入门教程和一条高信号高级案例。"),
        _tr(lang, "Every frequently asked question should become FAQ text, an example note, or a card field.", "每个高频问题都应沉淀为 FAQ、案例说明或 card 字段。"),
        _tr(lang, "Every report should be shareable and should teach the next action, not only describe the current data.", "每个报告都应既可分享，又能告诉用户下一步，而不只是描述当前数据。"),
        _tr(lang, "Every saved asset should reduce future support load by being self-explaining.", "每个保存资产都应该通过自解释来减少未来支持负担。"),
    ]
    body += _card(_tr(lang, 'Operational rules that scale', '真正能扩容的运营规则'), _bullets(capacity))
    body += _card(_tr(lang, 'The growth loop', '增长回路'), _bullets([
        _tr(lang, "People discover a tutorial.", "用户先发现教程。"),
        _tr(lang, "They copy an example and save a dataset asset.", "他们复制一个案例并保存数据资产。"),
        _tr(lang, "They share a linked EDA report or docs bundle.", "他们分享 linked EDA report 或 docs bundle。"),
        _tr(lang, "The shared asset becomes onboarding material for the next person.", "这个分享出去的资产又会成为下一个人的 onboarding 材料。"),
    ]))
    body += "</div></div>"
    body += f"<div class='section' id='loop'>" + _card(_tr(lang, 'Community feedback loop', '社区反馈回路'), _bullets([
        _tr(lang, "Track which examples are opened most often and expand those clusters first.", "跟踪哪些案例被打开得最多，优先扩这些案例簇。"),
        _tr(lang, "Track which tasks users derive most often from real-data EDA reports.", "跟踪用户从真实数据 EDA 报告最常走向哪些任务。"),
        _tr(lang, "Track which card fields and schema fields reduce support friction the most.", "跟踪哪些 card/schema 字段最能减少支持摩擦。"),
    ])) + "</div>"
    body += f"<div class='section' id='checklist'>" + _card(_tr(lang, 'Release checklist', '发布 checklist'), _bullets([
        _tr(lang, "Does the English-first docs surface have a working Chinese switch?", "英文优先文档表层是否有可用的中文切换？"),
        _tr(lang, "Can a new user get from landing to first saved dataset in under ten minutes?", "新用户能否在 10 分钟内从 landing 走到第一个保存好的数据集？"),
        _tr(lang, "Do examples cover forecasting, classification, real-data EDA, control/causal, agent assets, and publishing?", "案例是否覆盖 forecasting、classification、真实数据 EDA、control/causal、agent assets 和 publishing？"),
        _tr(lang, "Do linked EDA reports route users into the correct docs, examples, API pages, and FAQ entries?", "linked EDA report 是否能把用户导到正确的 docs、案例、API 页面和 FAQ？"),
    ])) + "</div>"
    return _page("TSDataForge Adoption", body, lang=lang, slug="rollout.html")



def _api_reference_page(lang: str, api_ref: APIReference) -> str:
    scenario_rows = []
    for item in scenario_catalog(language=lang):
        scenario_rows.append([
            escape(item.title),
            '<br/>'.join([f"<code>{escape(name)}</code>" for name in item.api_names[:5]]) or '—',
            '<br/>'.join([f"<code>{escape(_env_title(env_id, lang))}</code>" for env_id in item.environment_ids]) or '—',
            '<br/>'.join([f"<code>{escape(ex_id)}</code>" for ex_id in item.example_ids[:4]]) or '—',
        ])
    body = _hero(
        _tr(lang, 'API Reference: explain what each API is for, why it exists, and where it fits', 'API Reference：解释每个 API 是干什么的、为什么存在、以及适合放在哪个场景里'),
        _tr(lang, 'This is not only a symbol inventory. It starts from the curated public surface and only then descends into the rest of the library. Every category and every key symbol should answer four questions: what does it do, why does it exist, when should I use it, and which environment or scenario is it best in?', '这不只是符号清单，而是一张从用户意图走到正确 API 表层的公开地图。每个分类、每个关键符号都应该回答四个问题：它做什么、为什么存在、什么时候该用、最适合什么环境或场景。'),
        ['intent-first', 'why-driven', 'environment-aware', 'scenario-aware'],
        pills=[f'{api_ref.n_symbols} public symbols', f'{build_api_reference(mode="full").n_symbols} full symbols', f'{len(api_ref.categories)} categories'],
    )
    body += _toc([
        ('map', _tr(lang, 'Category map', '分类地图')),
        ('goal-map', _tr(lang, 'API by scenario', '按场景看的 API')), 
        ('eda-map', _tr(lang, 'EDA to API routing', '从 EDA 到 API 的路由')),
    ])
    body += f"<div class='section notice'><strong>{escape(_tr(lang, 'Ecosystem note', '生态说明'))}:</strong> {escape(_tr(lang, 'If you are still deciding whether TSDataForge is the main package or the surrounding layer in your workflow, read the positioning page first. The API surface becomes much clearer after that.', '如果你还在判断 TSDataForge 在你的 workflow 里是主包还是外围层，建议先看 positioning 页面。看完之后 API 表层会清晰很多。'))} <a href='positioning.html'>{escape(_tr(lang, 'Open positioning', '查看定位页'))}</a></div>"
    body += f"<div class='section' id='map'><div class='grid'>"
    for cat in api_ref.categories:
        body += _card(
            _api_category_title(cat, lang),
            f"<p>{escape(_api_category_summary(cat, lang))}</p>"
            f"<p><strong>{len(cat.symbols)}</strong> {escape(_tr(lang, 'public symbols', '公开符号'))}</p>"
            f"<p><a href='api/{escape(cat.category_id)}.html'>{escape(_tr(lang, 'Open category page', '打开分类页'))}</a></p>",
        )
    body += '</div></div>'
    body += f"<div class='section' id='goal-map'><div class='kicker'>Scenario map</div><h2>{escape(_tr(lang, 'Which APIs matter in which scenario', '不同场景下最重要的是哪些 API'))}</h2>{_table([_tr(lang, 'Scenario', '场景'), _tr(lang, 'Start with APIs', '先看 API'), _tr(lang, 'Best environments', '适合环境'), _tr(lang, 'Examples', '案例')], scenario_rows)}</div>"
    body += f"<div class='section' id='eda-map'><div class='kicker'>Routing</div><h2>{escape(_tr(lang, 'From EDA findings to public APIs', '从 EDA 发现走到公共 API'))}</h2>{_route_table(lang)}</div>"
    body += f"<div class='section'><div class='tip'><strong>{escape(_tr(lang, 'Reading tip', '阅读提示'))}:</strong> {escape(_tr(lang, 'Most users should not begin with primitive generators. Start with Workflow, Analysis, and Agent on the category page, then descend into operators and primitives only after the workflow is clear.', '大多数用户不应该从最底层 primitive generator 开始。建议先看 Workflow、Analysis 和 Agent 这些分类，再在工作流清楚之后下钻到 operators 和 primitives。'))}</div></div>"
    return _page('TSDataForge API Reference', body, lang=lang, slug='api-reference.html')

def _faq_page(lang: str) -> str:
    extra = [
        (_tr(lang, 'What is TSDataForge actually for?', 'TSDataForge 到底是拿来做什么的？'), _tr(lang, 'It is for defining time-series structure, explaining real data, deriving task datasets, and exporting self-explaining assets. The package matters most when you need those layers to stay connected instead of becoming separate scripts.', '它是用来定义时间序列结构、解释真实数据、派生任务数据集，并导出自解释资产的。只有当你需要这些层始终连在一起，而不是分裂成各自脚本时，这个包的价值才会最大。')),
        (_tr(lang, 'Where should a new user start?', '新用户应该从哪里开始？'), _tr(lang, 'Usually in a Jupyter notebook or a small Python script. Start with one of two paths: one generated series, or one real-data EDA report. Only then move to datasets, taskification, and publication assets.', '通常从 Jupyter notebook 或一个小的 Python 脚本开始。先走两条路径中的一条：生成一条序列，或者做一份真实数据 EDA 报告。之后再进入数据集、任务化和发布资产。')),
        (_tr(lang, 'Why not just use sktime, Darts, tsfresh, tslearn, GluonTS, aeon, STUMPY, or YData Profiling?', '为什么不直接用 sktime、Darts、tsfresh、tslearn、GluonTS、aeon、STUMPY 或 YData Profiling？'), _tr(lang, 'Because those libraries are usually strongest at a narrower slice: estimators, forecasting models, feature extraction, matrix-profile mining, or generic profiling. TSDataForge focuses on the asset layer that connects raw or simulated sequence data to explanation, taskification, and low-token handoff. In many serious projects, the best answer is to use both.', '因为这些库通常在更窄的一段上最强：estimator、forecasting model、feature extraction、matrix-profile mining 或通用 profiling。TSDataForge 更关注的是把原始或仿真的序列数据接到 explanation、taskification 和低 token handoff 上的资产层。在很多严肃项目里，最好的答案其实是同时使用两者。')),
    ]
    body = _hero(
        'FAQ',
        _tr(lang, 'This page answers both repeated support questions and the framing questions that determine whether users understand the package at all.', '这页既回答重复出现的支持问题，也回答那些决定用户是否真正理解这个包的 framing 问题。'),
        ['support', 'framing', 'EDA-linked'],
        pills=['new users', 'maintainers', 'agents'],
    )
    toc_items = [('package-purpose', extra[0][0]), ('where-start', extra[1][0]), ('why-not-others', extra[2][0])] + [(item.faq_id, FAQ_TRANSLATIONS.get(item.faq_id, (item.question, item.question))[0 if not lang.startswith('zh') else 1]) for item in FAQ_ENTRIES]
    body += _toc(toc_items)
    body += f"<div class='section'>{_card(_tr(lang, 'How to use this page', '如何使用这页'), '<p>' + escape(_tr(lang, 'If you are new, read the first two framing questions before the support details. If you arrived from an EDA report, jump to the linked FAQ entries directly.', '如果你是新用户，先看最前面的两个 framing 问题，再看支持细节。如果你是从 EDA 报告跳过来的，就直接看报告里链接的 FAQ 条目。')) + '</p>')}{_route_table(lang)}</div>"
    body += f"<div class='section' id='package-purpose'>" + _card(extra[0][0], f"<p>{escape(extra[0][1])}</p>") + '</div>'
    body += f"<div class='section' id='where-start'>" + _card(extra[1][0], f"<p>{escape(extra[1][1])}</p>") + '</div>'
    body += f"<div class='section' id='why-not-others'>" + _card(extra[2][0], f"<p>{escape(extra[2][1])}</p><p><a href='positioning.html'>{escape(_tr(lang, 'Open the positioning page', '查看定位页'))}</a></p>") + '</div>'
    for item in FAQ_ENTRIES:
        q_en, q_zh = FAQ_TRANSLATIONS.get(item.faq_id, (item.question, item.question))
        answer = item.answer if not lang.startswith('zh') else FAQ_ANSWERS_ZH.get(item.faq_id, item.answer)
        body += f"<div class='section' id='{escape(item.faq_id)}'>" + _card(_tr(lang, q_en, q_zh), f"<p>{escape(answer)}</p>") + '</div>'
    return _page('TSDataForge FAQ', body, lang=lang, slug='faq.html')

def _landing_page_refined(lang: str, catalog: list[ExampleRecipe], tutorials: list[TutorialTrack], api_ref: APIReference) -> str:
    matrix = build_positioning_matrix(language=lang)
    featured_examples = recommend_examples('quickstart eda taskify handoff api', top_k=6, language=lang)
    hot_examples = recommend_examples('openclaw github stars bitcoin gold oil market similarity live', top_k=4, language=lang)
    featured_tutorials = recommend_tutorials('quickstart real data handoff launch', top_k=4, language=lang)
    featured_scenarios = recommend_scenarios('real data benchmark control causal docs adoption', top_k=6, language=lang)
    featured_envs = recommend_environments('notebook script agent docs release', top_k=4, language=lang)
    featured_playbooks = recommend_playbooks('first success real data handoff benchmark', top_k=3, language=lang)
    featured_starters = recommend_starters('new user real data handoff benchmark', top_k=3, language=lang)

    body = _hero(
        _tr(lang, 'Turn raw time-series datasets into profiling reports, handoff bundles, and clear next steps', '把原始时间序列数据集变成 profiling report、handoff bundle 和明确下一步'),
        _tr(lang, 'Use one command or one function call to understand a raw time-series dataset before model selection or handoff.', '用一条命令或一次函数调用，在选模型或交接前先理解原始时序数据集。'),
        ['time-series profiling', 'handoff bundles', 'real public demos', 'dataset cards', 'schema-first'],
        pills=[f'{len(catalog)} examples', f'{len(tutorials)} tutorial tracks', f'{api_ref.n_symbols} public symbols'],
    )
    body += _home_identity_strip(lang)
    body += _toc([
        ('what', _tr(lang, 'What the package is', '这个包是做什么的')),
        ('why', _tr(lang, 'Why it exists', '为什么会有这个包')),
        ('handoff', _tr(lang, 'The shortest happy path', '最短 happy path')),
        ('your-data', _tr(lang, 'Use your own data', 'Use your own data')),
        ('flagship', _tr(lang, 'Three real public demos', '三个真实公开数据案例')),
        ('surface', _tr(lang, 'The five APIs to remember', '最该记住的五个 API')),
        ('positioning', _tr(lang, 'How it differs from other libraries', '它与其他库的差异')),
        ('jobs', _tr(lang, 'What you can do with it', '你可以拿它做什么')),
        ('environments', _tr(lang, 'Which environment should I use', '我应该在哪种环境里使用它')),
        ('api-map', _tr(lang, 'API map by intent', '按意图看的 API 地图')),
        ('examples', _tr(lang, 'Open these first', '先打开这些内容')),
        ('hot-now', _tr(lang, 'Public data examples', '公共数据案例')),
    ])
    body += f"<div class='section' id='flagship'><div class='kicker'>Real public demos</div><h2>{escape(_tr(lang, 'Start here: three real public demos', '先看这里：三个真实公开数据案例'))}</h2><div class='grid'>"
    for spec in [item for item in _showcase_specs(lang) if item["group"] == "public"][:3]:
        body += _showcase_card(spec, lang=lang, include_showcase_link=True)
    body += '</div></div>'
    body += f"<div class='section' id='what'><div class='grid'>"
    body += _card(_tr(lang, 'What TSDataForge is', 'TSDataForge 是什么'), '<p>' + escape(_tr(lang, 'It is a time-series profiling and handoff layer: load data, explain it, package it, and route it into the next task.', '它是一个时序数据 profiling 与 handoff 层：载入数据、解释数据、打包数据，并把它路由到下一步任务。')) + '</p>')
    body += _card(_tr(lang, 'What it is not', '它不是什么'), _bullets([
        _tr(lang, 'It is not a forecasting model zoo.', '它不是 forecasting 模型库。'),
        _tr(lang, 'It is not a replacement for domain simulators or heavy analytics stacks.', '它不是领域仿真器或大型分析栈的替代品。'),
        _tr(lang, 'It is not only for synthetic data; understanding real data is a first-class workflow.', '它也不只针对合成数据；理解真实数据同样是一等工作流。'),
    ]))
    body += _card(_tr(lang, 'Who it is for', '它是给谁用的'), _bullets([
        _tr(lang, 'Researchers who need to understand a raw dataset before choosing the task.', '需要在选任务前先理解原始数据集的研究人员。'),
        _tr(lang, 'Applied teams who need a shareable handoff bundle for teammates, students, or clients.', '需要给同事、学生或客户交接数据 bundle 的应用团队。'),
        _tr(lang, 'Automation or tooling builders who need compact, schema-stable interfaces.', '需要 compact 且 schema 稳定接口的自动化或工具链构建者。'),
    ]))
    body += '</div></div>'
    body += f"<div class='section' id='why'><div class='grid'>"
    body += _card(_tr(lang, 'Why this package exists', '为什么会有这个包'), '<p>' + escape(_tr(lang, 'Time-series work often starts with raw files and unclear task boundaries. TSDataForge exists to turn that first-contact moment into a profiling report, a handoff bundle, and an explicit next-step record.', '时间序列工作常常从原始文件和不清晰的任务边界开始。TSDataForge 的存在就是把这个第一接触时刻变成 profiling report、handoff bundle 和明确的下一步记录。')) + '</p>')
    body += _card(_tr(lang, 'The design bet', '核心设计判断'), _bullets([
        _tr(lang, 'Real data should be described before it is modeled.', '真实数据应该先被描述，再被建模。'),
        _tr(lang, 'One base dataset should power many downstream tasks.', '一份基础数据集应该服务多种下游任务。'),
        _tr(lang, 'Saved artifacts should carry schema, context, cards, and reports.', '保存下来的产物应该自带 schema、context、card 和报告。'),
    ]))
    body += _card(_tr(lang, 'The simplest way to think about it', '最简单的理解方式'), _bullets([
        'load_asset -> report',
        'report -> handoff bundle',
        'handoff -> choose next task',
        'taskify -> train or benchmark',
    ]))
    body += '</div></div>'
    body += f"<div class='section' id='handoff'><div class='kicker'>Happy path</div><h2>{escape(_tr(lang, 'Start with one handoff bundle, not a long module tour', '先生成一个 handoff bundle，而不是先看很长的模块列表'))}</h2><div class='two-col'>"
    body += _card(_tr(lang, 'Recommended first call', '推荐第一条调用'), "<pre><code>from tsdataforge import handoff\n\nbundle = handoff(\n    'my_dataset.npy',\n    output_dir='dataset_handoff_bundle',\n    dataset_id='lab_values',\n    include_schemas=True,\n)\nprint(bundle.output_dir)</code></pre><p class='small'>" + escape(_tr(lang, 'This creates report + context + card + handoff index + manifest + schemas in one predictable directory.', '这会把 report、context、card、handoff index、manifest 和 schemas 放进一个可预测目录。')) + "</p>")
    body += _card(_tr(lang, 'Open in this order', '推荐打开顺序'), _bullets([
        _tr(lang, '1. report.html', '1. report.html'),
        _tr(lang, '2. dataset_card.md', '2. dataset_card.md'),
        _tr(lang, '3. dataset_context.json', '3. dataset_context.json'),
        _tr(lang, '4. handoff_index_min.json', '4. handoff_index_min.json'),
        _tr(lang, '5. choose one next action from the bundle', '5. 从 bundle 里挑一个 next action'),
    ]))
    body += "</div></div>"
    body += _bring_your_own_data_section(lang, section_id='your-data')
    body += f"<div class='section' id='surface'><div class='kicker'>Public surface</div><h2>{escape(_tr(lang, 'The five APIs to remember first', '最先记住的五个 API'))}</h2><p class='muted'>{escape(_tr(lang, 'Everything else in TSDataForge can stay advanced for a while. These five entry points are the public product surface the README, docs, and agents should agree on.', 'TSDataForge 里的其他东西暂时都可以留在 advanced 区。这五个入口就是 README、docs 和 agents 应该共同遵守的公共产品表层。'))}</p>{_public_surface_table(lang)}</div>"
    top_companions = [item for item in matrix.profiles if item.kind != 'self'][:4]
    comp_rows = []
    for item in top_companions:
        diff = item.tsdataforge_difference if not lang.startswith('zh') else (item.tsdataforge_difference_zh or item.tsdataforge_difference)
        comp_rows.append([f"<a href='{escape(item.official_url)}'><strong>{escape(item.title)}</strong></a>", escape('; '.join(item.best_for[:2]) or '-'), escape(diff)])
    body += f"<div class='section' id='positioning'><div class='kicker'>Positioning</div><h2>{escape(_tr(lang, 'Why this is not just another time-series package', '为什么这不是又一个普通时间序列包'))}</h2><p class='muted'>{escape(_tr(lang, 'The adoption question is whether people can tell, in one screen, when TSDataForge is the right layer and when another library should take over next.', '真正的采用问题是，用户能不能在一个屏幕内看懂：什么时候 TSDataForge 是对的那一层，接下来又该由哪个库接手。'))}</p>{_table([_tr(lang, 'Library', '库'), _tr(lang, 'Usually strongest at', '通常最擅长'), _tr(lang, 'TSDataForge difference', 'TSDataForge 的差异')], comp_rows)}<p><a href='positioning.html'>{escape(_tr(lang, 'Open the full ecosystem-fit page', '打开完整生态定位页'))}</a></p></div>"
    body += f"<div class='section' id='jobs'><div class='kicker'>Capabilities</div><h2>{escape(_tr(lang, 'What you can do with it', '你可以用这个包做什么'))}</h2>{_scenario_cards(featured_scenarios, lang=lang)}</div>"
    body += f"<div class='section' id='environments'><div class='kicker'>Environments</div><h2>{escape(_tr(lang, 'Which environment should I use', '我应该在哪种环境里使用它'))}</h2><p class='muted'>{escape(_tr(lang, 'Start in a notebook for exploration, move to scripts for reproducible assets, use CI for docs checks, and use compact contexts when an agent is in the loop.', '探索先用 notebook，产出可复现资产用脚本，文档检查用 CI，而一旦接入 agent 就优先使用 compact context。'))}</p>{_environment_cards(featured_envs, lang=lang)}</div>"
    workflow_rows = [
        [_tr(lang, 'I have raw data and do not know the task yet', '我手里有原始数据，还不知道该做什么任务'), '<code>load_asset</code> / <code>report</code>', '<code>real_series_eda</code>'],
        [_tr(lang, 'I want one reusable bundle for handoff', '我想要一个可复用的 handoff bundle'), '<code>handoff</code>', '<code>dataset_handoff_bundle</code>'],
        [_tr(lang, 'I want one base dataset for many tasks', '我想让一份基础数据集服务多种任务'), '<code>load_asset</code> / <code>taskify</code>', '<code>taskify_forecasting</code>'],
        [_tr(lang, 'I want a public demo quickly', '我想快速跑一个公开 demo'), '<code>demo</code>', '<code>quickstart</code>'],
    ]
    body += f"<div class='section' id='api-map'><div class='kicker'>API map</div><h2>{escape(_tr(lang, 'Start from intent, not from modules', '从意图进入，而不是从模块猜'))}</h2>{_table([_tr(lang, 'What you want to do', '你要做什么'), _tr(lang, 'Start with APIs', '先看哪些 API'), _tr(lang, 'Start with example', '先看哪个案例')], workflow_rows)}</div>"
    body += f"<div class='section' id='examples'><div class='grid'>" + _card(_tr(lang, 'Open these examples first', '建议先打开的案例'), _example_cards(featured_examples, lang=lang)) + _card(_tr(lang, 'Follow these tutorial tracks', '建议先走的教程路径'), _tutorial_cards(featured_tutorials, lang=lang)) + "</div></div>"
    body += f"<div class='section' id='hot-now'><div class='kicker'>Public data</div><h2>{escape(_tr(lang, 'Public data examples that are easy to share', '容易分享的公共数据案例'))}</h2><p class='muted'>{escape(_tr(lang, 'These examples use public signals such as GitHub attention, crypto, gold, and oil. They work as both tutorials and demos.', '这些案例使用 GitHub 热度、加密货币、黄金和原油等公共信号，既能教学，也能演示。'))}</p>{_example_cards(hot_examples, lang=lang)}</div>"
    body += f"<div class='section'><div class='grid'>" + _card(_tr(lang, 'Choose a workflow if you care more about the goal than the modules', '如果你更关心目标而不是模块名，就先选一条工作流'), _playbook_cards(featured_playbooks, lang=lang)) + _card(_tr(lang, 'Open a starter project if you want a ready project layout', '如果你想直接拿到项目骨架，就先打开起步项目'), _starter_cards(featured_starters, lang=lang)) + "</div></div>"
    body += f"<div class='section'><div class='notice'><strong>{escape(_tr(lang, 'If you only remember one thing', '如果你只记住一件事'))}:</strong> {escape(_tr(lang, 'Think in this order: understand the data, save the handoff bundle, choose the task, then move into the modeling library.', '按照这个顺序思考：先理解数据，再保存 handoff bundle，再选择任务，最后进入建模库。'))}</div></div>"
    body = body.replace('Public data examples that are easy to share', 'Hot right now: public data examples that are easy to share')
    return _page('TSDataForge Docs', body, lang=lang, slug='index.html')


def _getting_started_page_refined(lang: str) -> str:
    quick_examples = recommend_examples('quickstart dataset taskify eda', top_k=4, language=lang)
    envs = recommend_environments('notebook script real data first success', top_k=3, language=lang)
    body = _hero(
        _tr(lang, 'Quickstart: understand what the package does, then get the first meaningful success in five minutes', '快速上手：先搞清这个包做什么，再在 5 分钟内跑通第一次有意义的成功'),
        _tr(lang, 'Do not start by memorizing modules. Start by generating one profiling report or one handoff bundle and open the saved files in order.', '不要先背模块名。先生成一份 profiling report 或一个 handoff bundle，然后按顺序打开产物。'),
        ['load_asset', 'report', 'handoff', 'taskify', 'demo'],
        pills=['copy/paste', 'first success', 'profiling-first'],
    )
    body += _toc([
        ('what', _tr(lang, 'What this package is for', '这个包是干什么的')),
        ('surface', _tr(lang, 'The five APIs to learn first', '先学会的五个 API')),
        ('your-data', _tr(lang, 'Use your own data', 'Use your own data')),
        ('ladder', _tr(lang, '30-second / 5-minute / 20-minute ladder', '30 秒 / 5 分钟 / 20 分钟路径')),
        ('five', _tr(lang, 'The first four code blocks', '前四段代码')),
        ('envs', _tr(lang, 'Good starting environments', '最合适的起步环境')),
        ('pitfalls', _tr(lang, 'Pitfalls', '常见坑')),
    ])
    body += f"<div class='section' id='what'><div class='grid'>"
    body += _card(_tr(lang, 'The one-sentence answer', '一句话回答'), '<p>' + escape(_tr(lang, 'TSDataForge helps you understand a raw time-series dataset, produce a profiling report, package a handoff bundle, and only then convert the data into downstream tasks.', 'TSDataForge 帮你理解原始时间序列数据集、产出 profiling report、打包 handoff bundle，然后再把数据转成下游任务。')) + '</p>')
    body += _card(_tr(lang, 'If you already have real data', '如果你已经有真实数据'), '<p>' + escape(_tr(lang, 'Do not start by picking a model. Start with `report(...)` or `handoff(...)`.', '不要先选模型，先从 `report(...)` 或 `handoff(...)` 开始。')) + '</p>')
    body += _card(_tr(lang, 'If you want a benchmark', '如果你想做 benchmark'), '<p>' + escape(_tr(lang, 'Do not start from `generate_dataset` unless the task is fixed. Start from a reusable base dataset whenever you can.', '除非任务已经固定，否则不要一上来就用 `generate_dataset`。能从可复用基础数据集开始就尽量从那里开始。')) + '</p>')
    body += '</div></div>'
    body += f"<div class='section' id='surface'><h2>{escape(_tr(lang, 'The five APIs to learn first', '最先要学会的五个 API'))}</h2>{_public_surface_cards(lang)}</div>"
    body += _bring_your_own_data_section(lang, section_id='your-data')
    body += f"<div class='section' id='ladder'><div class='tip'><strong>{escape(_tr(lang, '60-second demo', '60 秒演示'))}:</strong><pre><code>git clone https://github.com/ZipengWu365/TSDataForge.git\ncd TSDataForge\npip install &quot;.[viz]&quot;\npython -m tsdataforge demo --output demo_bundle</code></pre><p>{escape(_tr(lang, 'Open `demo_bundle/report.html` first.', '先打开 `demo_bundle/report.html`。'))}</p></div></div>"
    blocks = [
        (
            _tr(lang, '1) Create the first profiling report', '1）生成第一份 profiling report'),
            '''from tsdataforge import load_asset, report

dataset = load_asset('demo.npy')
report(dataset, output_path='report.html')''',
            _tr(lang, 'If you already have a file, the first public move is simply `load_asset(...)` followed by `report(...)`.', '如果你已经有文件，第一条公开路径就是 `load_asset(...)` 然后接 `report(...)`。'),
        ),
        (
            _tr(lang, '2) Build the handoff bundle', '2）生成 handoff bundle'),
            '''from tsdataforge import handoff

bundle = handoff(
    'demo.npy',
    output_dir='dataset_handoff_bundle',
)
print(bundle.output_dir)''',
            _tr(lang, 'This is the shortest outcome-first path: one call yields report + context + card + index + next actions.', '这是最短的 outcome-first 路径：一个调用就能得到 report、context、card、index 和 next actions。'),
        ),
        (
            _tr(lang, '3) Open the report and pick the next task', '3）打开 report 并决定下一个任务'),
            '''# open dataset_handoff_bundle/report.html first
# then read dataset_card.md and handoff_index_min.json''',
            _tr(lang, 'Use the bundle to decide whether forecasting, anomaly, change point, or control/causal routing makes sense.', '借助 bundle 决定接下来走 forecasting、anomaly、change point，还是 control/causal。'),
        ),
        (
            _tr(lang, '4) Taskify only after the report', '4）看完 report 再任务化'),
            '''from tsdataforge import load_asset, taskify

base = load_asset('demo.npy')
forecast = taskify(base, task='forecasting', horizon=24)
forecast.save('saved_forecast_dataset')''',
            _tr(lang, 'Now the output is trainable, explainable, and shareable.', '现在这个输出既可训练、可解释，也可分享。'),
        ),
    ]
    body += f"<div class='section' id='five'><h2>{escape(_tr(lang, 'The first four code blocks to run', '最先应该跑通的四段代码'))}</h2><div class='grid'>"
    for title, code, desc in blocks:
        body += _card(title, f"<p>{escape(desc)}</p><pre><code>{escape(code)}</code></pre>")
    body += '</div></div>'
    body += f"<div class='section' id='envs'><h2>{escape(_tr(lang, 'Good starting environments', '最合适的起步环境'))}</h2>{_environment_cards(envs, lang=lang)}</div>"
    body += f"<div class='section'><h2>{escape(_tr(lang, 'Choose your next example by goal', '按目标选择下一条案例'))}</h2>{_example_cards(quick_examples, lang=lang)}</div>"
    body += f"<div class='section' id='pitfalls'><div class='warning'><strong>{escape(_tr(lang, 'Common pitfall', '常见坑'))}:</strong> {escape(_tr(lang, 'Do not force every dataset into forecasting on day one. For real data, the best first step is usually describe + EDA, not model selection.', '不要在第一天就把所有数据都强行做 forecasting。面对真实数据时，最好的第一步通常是 describe + EDA，而不是先选模型。'))}</div></div>"
    return _page('TSDataForge Quickstart', body, lang=lang, slug='getting-started.html')


def _handoff_page_refined(lang: str) -> str:
    body = _hero(
        _tr(lang, 'Handoff: the shortest dataset -> report -> next-action path', '交接路径：dataset -> report -> next action 的最短路径'),
        _tr(lang, 'Generate one handoff bundle when you need a profiling report, a dataset card, a compact context, and an explicit next step in one predictable directory.', '当你需要把 profiling report、dataset card、compact context 和明确下一步放进同一个可预测目录时，就生成一个 handoff bundle。'),
        ['report-first', 'profiling', 'shareable artifacts'],
        pills=['handoff', 'CLI', 'README-ready'],
    )
    body += _toc([
        ('why', _tr(lang, 'Why start here', '为什么应该从这里开始')),
        ('your-data', _tr(lang, 'Use your own data', 'Use your own data')),
        ('python', _tr(lang, 'One Python call', '一个 Python 调用')),
        ('open-order', _tr(lang, 'Human and agent open order', '人类和 agent 的打开顺序')),
        ('actions', _tr(lang, 'Why the next-action plan is believable', '为什么 next-action plan 是可信的')),
        ('cli', _tr(lang, 'One CLI command', '一个 CLI 命令')),
        ('artifacts', _tr(lang, 'What the bundle contains', 'bundle 里有什么')),
        ('when', _tr(lang, 'When to use it', '什么时候用')),
    ])
    body += f"<div class='section' id='why'><div class='grid'>"
    body += _card(_tr(lang, 'What problem it solves', '它解决什么问题'), '<p>' + escape(_tr(lang, 'Most users do not need a long API tour on day one. They need one outcome: a profiling report that is already packaged for the next person, script, or agent.', '大多数用户在第一天并不需要一长串 API 导览，他们需要的是一个结果：一份已经适合交给下一个人、脚本或 agent 的 profiling report。')) + '</p>')
    body += _card(_tr(lang, 'What makes it different', '它和单独的 report / card 有什么不同'), '<p>' + escape(_tr(lang, 'The handoff bundle is not one more artifact. It deliberately packages report + context + card + manifest + next actions into one predictable directory.', 'handoff bundle 不是再多一个 artifact，而是把 report、context、card、manifest 和 next actions 有意识地打包成一个可预测目录。')) + '</p>')
    body += _card(_tr(lang, 'What to avoid', '要避免什么'), '<p>' + escape(_tr(lang, 'Avoid starting by pasting raw arrays into prompts or sending only a long README. Start with the bundle, then open the raw asset only if needed.', '避免一开始就把原始数组塞进 prompt，或者只给一份很长的 README。应该先从 bundle 开始，只有在必要时再打开原始资产。')) + '</p>')
    body += '</div></div>'
    body += _bring_your_own_data_section(lang, section_id='your-data')
    body += f"<div class='section' id='open-order'><h2>{escape(_tr(lang, 'Human and agent open order', '人类和 agent 的打开顺序'))}</h2><div class='grid'>"
    body += _card(_tr(lang, 'Human open order', '人类打开顺序'), "<ol><li><code>report.html</code></li><li><code>dataset_card.md</code></li><li><code>dataset_context.json</code></li><li><code>handoff_index_min.json</code></li><li><code>action_plan.json</code></li></ol>")
    body += _card(_tr(lang, 'Agent open order', 'agent 打开顺序'), "<ol><li><code>handoff_index_min.json</code></li><li><code>dataset_context.json</code></li><li><code>dataset_card.md</code></li><li><code>action_plan.json</code></li><li><code>recommended_next_step</code></li></ol><p class='small'>" + escape(_tr(lang, 'Do not open `handoff_bundle.json` unless a required field is missing.', '除非缺少必要字段，否则不要先打开 `handoff_bundle.json`。')) + "</p>")
    body += '</div></div>'
    body += f"<div class='section' id='actions'><div class='notice'><strong>{escape(_tr(lang, 'What changed in the action plan', 'next-action plan 做了什么改进'))}:</strong> {escape(_tr(lang, 'The bundle separates already-done work from the recommended next step, so the first non-open action no longer tells you to rebuild the bundle you already have.', 'bundle 现在会把 already-done 和 recommended next step 分开，因此第一条非打开动作不会再让你重建已经存在的 bundle。'))}</div></div>"
    body += f"<div class='section' id='python'><h2>{escape(_tr(lang, 'One Python call', '一个 Python 调用'))}</h2>"
    body += _card(
        _tr(lang, 'Recommended happy path', '推荐 happy path'),
        "<pre><code>from tsdataforge import load_asset, handoff\n\nbase = load_asset('my_dataset.npy')\n\nbundle = handoff(\n    base,\n    output_dir='dataset_handoff_bundle',\n    include_schemas=True,\n)\nprint(bundle.output_dir)</code></pre>"
        + "<p class='small'>"
        + escape(_tr(lang, 'Open `report.html` first. Then read `dataset_card.md` and `dataset_context.json` before picking the next task.', '先打开 `report.html`，再读 `dataset_card.md` 和 `dataset_context.json`，最后再决定下一个任务。'))
        + "</p>"
    )
    body += '</div>'
    body += f"<div class='section' id='cli'><h2>{escape(_tr(lang, 'One CLI command', '一个 CLI 命令'))}</h2>"
    body += _card(
        _tr(lang, 'CLI for saved arrays or external assets', '面向保存数组或外部资产的 CLI'),
        "<pre><code>tsdataforge handoff my_dataset.npy --output handoff_bundle\ntsdataforge report my_dataset.npy --output report.html</code></pre>"
        + "<p class='small'>"
        + escape(_tr(lang, 'Use the CLI when you want a shortest path for teammates who should not have to write Python glue first.', '当你希望队友不必先写 Python glue 就能跑通最短路径时，用 CLI。'))
        + "</p>"
    )
    body += '</div>'
    rows = [
        ['report.html', _tr(lang, 'Outcome-first HTML EDA report', '面向结果的 HTML EDA 报告')],
        ['dataset_context.json / .md', _tr(lang, 'Compact context for prompts and agent handoff', '给 prompt 和 agent handoff 用的紧凑 context')],
        ['dataset_card.json / .md', _tr(lang, 'Human + machine-readable asset summary', '可供人和机器同时读取的资产摘要')],
        ['handoff_index_min.json / .md', _tr(lang, 'Smallest first-entry contract', '最小的 first-entry 契约')],
        ['action_plan.json / .md', _tr(lang, 'Detailed already_done / recommended / optional plan', '详细 already_done / recommended / optional 计划')],
        ['manifest / handoff_bundle', _tr(lang, 'One predictable inventory of what was saved and what to do next', '把保存内容和下一步动作集中到一个可预测清单里')],
        ['asset/', _tr(lang, 'Optional raw dataset export', '可选的原始数据集导出')],
    ]
    body += f"<div class='section' id='artifacts'><h2>{escape(_tr(lang, 'What the bundle contains', 'bundle 里有什么'))}</h2>{_table([_tr(lang, 'Artifact', '产物'), _tr(lang, 'Why it is there', '为什么在这里')], rows)}</div>"
    body += f"<div class='section' id='when'><h2>{escape(_tr(lang, 'Use it first when ...', '在这些情况下应优先使用'))}</h2>"
    body += _card(
        _tr(lang, 'Good fit', '适合场景'),
        _bullets([
            _tr(lang, 'You have a real dataset and need the first shareable explanation artifact.', '你拿到真实数据，想先做第一份可分享解释产物。'),
            _tr(lang, 'You want to hand a dataset to another researcher without losing task semantics.', '你想把数据交给另一位研究者，同时不丢掉任务语义。'),
            _tr(lang, 'An agent should read a compact context instead of raw arrays.', '你希望 agent 先读 compact context，而不是原始数组。'),
            _tr(lang, 'You want a repeatable public demo for GitHub or docs.', '你想为 GitHub 或 docs 准备一条可重复的公开演示路径。'),
        ])
    )
    body += '</div>'
    return _page('TSDataForge Handoff', body, lang=lang, slug='handoff.html')


def generate_docs_site(output_dir: str | Path, *, title: str = "TSDataForge Docs") -> DocsSiteResult:
    out = Path(output_dir)
    examples_dir = out / "examples"
    api_dir = out / "api"
    notebooks_dir = out / "notebooks"
    starters_dir = out / "starters"
    showcase_dir = out / "showcase-bundles"
    zh_dir = out / "zh"
    zh_examples_dir = zh_dir / "examples"
    zh_api_dir = zh_dir / "api"
    zh_notebooks_dir = zh_dir / "notebooks"
    zh_starters_dir = zh_dir / "starters"
    for d in (examples_dir, api_dir, notebooks_dir, starters_dir, showcase_dir, zh_examples_dir, zh_api_dir, zh_notebooks_dir, zh_starters_dir):
        d.mkdir(parents=True, exist_ok=True)

    result = DocsSiteResult(output_dir=str(out))

    api_ref = build_api_reference()
    en_examples = example_catalog(language="en")
    zh_examples = example_catalog(language="zh")
    en_grouped = examples_by_category(language="en")
    zh_grouped = examples_by_category(language="zh")
    en_tutorials = tutorial_catalog(language="en")
    zh_tutorials = tutorial_catalog(language="zh")

    en_pages = {
        "index.html": _landing_page_refined("en", en_examples, en_tutorials, api_ref),
        "getting-started.html": _getting_started_page_refined("en"),
        "handoff.html": _handoff_page_refined("en"),
        "showcase.html": _showcase_page("en"),
        "tutorials.html": _tutorials_page("en"),
        "playbooks.html": _playbooks_page("en"),
        "starter-kits.html": _starter_kits_page("en"),
        "positioning.html": _positioning_page("en", build_positioning_matrix(language="en")),
        "cookbook.html": _cookbook_page_human("en", en_grouped),
        "taskification.html": _taskification_page("en"),
        "agent-playbook.html": _agent_playbook_page("en"),
        "use-cases.html": _use_cases_page("en"),
        "rollout.html": _rollout_page("en"),
        "api-reference.html": _api_reference_page("en", api_ref),
        "faq.html": _faq_page("en"),
    }
    zh_pages = {
        "index.html": _landing_page_refined("zh", zh_examples, zh_tutorials, api_ref),
        "getting-started.html": _getting_started_page_refined("zh"),
        "handoff.html": _handoff_page_refined("zh"),
        "showcase.html": _showcase_page("zh"),
        "tutorials.html": _tutorials_page("zh"),
        "playbooks.html": _playbooks_page("zh"),
        "starter-kits.html": _starter_kits_page("zh"),
        "positioning.html": _positioning_page("zh", build_positioning_matrix(language="zh")),
        "cookbook.html": _cookbook_page_human("zh", zh_grouped),
        "taskification.html": _taskification_page("zh"),
        "agent-playbook.html": _agent_playbook_page("zh"),
        "use-cases.html": _use_cases_page("zh"),
        "rollout.html": _rollout_page("zh"),
        "api-reference.html": _api_reference_page("zh", api_ref),
        "faq.html": _faq_page("zh"),
    }

    for name, html in en_pages.items():
        p = out / name
        p.write_text(html.replace("TSDataForge Docs", title), encoding="utf-8")
        result.pages.append(str(p))
    result.faq_page = str(out / "faq.html")

    for name, html in zh_pages.items():
        p = zh_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html.replace("TSDataForge Docs", title), encoding="utf-8")
        result.zh_pages.append(str(p))

    search_items: list[dict] = []
    for ex in en_examples:
        p = examples_dir / f"{ex.example_id}.html"
        p.write_text(_example_page_human(ex, lang="en"), encoding="utf-8")
        result.example_pages.append(str(p))
        search_items.append({
            "kind": "example",
            "lang": "en",
            "id": ex.example_id,
            "title": ex.title,
            "summary": ex.summary,
            "goal": ex.goal,
            "keywords": list(ex.keywords),
            "category": ex.category,
            "path": f"examples/{ex.example_id}.html",
        })
    for ex in zh_examples:
        p = zh_examples_dir / f"{ex.example_id}.html"
        p.write_text(_example_page_human(ex, lang="zh"), encoding="utf-8")
        result.example_pages.append(str(p))
        search_items.append({
            "kind": "example",
            "lang": "zh",
            "id": ex.example_id,
            "title": ex.title,
            "summary": ex.summary,
            "goal": ex.goal,
            "keywords": list(ex.keywords),
            "category": ex.category,
            "path": f"zh/examples/{ex.example_id}.html",
        })

    for cat in api_ref.categories:
        p = api_dir / f"{cat.category_id}.html"
        p.write_text(_api_category_page(cat, lang="en"), encoding="utf-8")
        result.api_pages.append(str(p))
        pzh = zh_api_dir / f"{cat.category_id}.html"
        pzh.write_text(_api_category_page(cat, lang="zh"), encoding="utf-8")
        result.api_pages.append(str(pzh))
        search_items.append({
            "kind": "api_category",
            "lang": "en",
            "id": cat.category_id,
            "title": _api_category_title(cat, "en"),
            "summary": _api_category_summary(cat, "en"),
            "path": f"api/{cat.category_id}.html",
            "keywords": [cat.category_id],
        })
        search_items.append({
            "kind": "api_category",
            "lang": "zh",
            "id": cat.category_id,
            "title": _api_category_title(cat, "zh"),
            "summary": _api_category_summary(cat, "zh"),
            "path": f"zh/api/{cat.category_id}.html",
            "keywords": [cat.category_id],
        })
        for sym in cat.symbols:
            search_items.append({
                "kind": "api_symbol",
                "lang": "en",
                "id": sym.name,
                "title": sym.name,
                "summary": _api_symbol_summary(sym, "en"),
                "path": f"api/{cat.category_id}.html#{sym.name}",
                "keywords": [sym.kind, cat.category_id, *sym.related, *sym.example_ids, *sym.works_in, *sym.scenario_ids],
            })
            search_items.append({
                "kind": "api_symbol",
                "lang": "zh",
                "id": sym.name,
                "title": sym.name,
                "summary": _api_symbol_summary(sym, "zh"),
                "path": f"zh/api/{cat.category_id}.html#{sym.name}",
                "keywords": [sym.kind, cat.category_id, *sym.related, *sym.example_ids, *sym.works_in, *sym.scenario_ids],
            })

    for slug, en, zh in NAV_ITEMS:
        search_items.append({"kind": "page", "lang": "en", "title": en, "path": slug, "keywords": []})
        search_items.append({"kind": "page", "lang": "zh", "title": zh, "path": f"zh/{slug}", "keywords": []})

    for item in scenario_catalog(language="en"):
        search_items.append({
            "kind": "scenario",
            "lang": "en",
            "id": item.scenario_id,
            "title": item.title,
            "summary": item.one_liner,
            "path": "use-cases.html#scenarios",
            "keywords": [*item.keywords, *item.api_names, *item.environment_ids],
        })
    for item in scenario_catalog(language="zh"):
        search_items.append({
            "kind": "scenario",
            "lang": "zh",
            "id": item.scenario_id,
            "title": item.title,
            "summary": item.one_liner,
            "path": "zh/use-cases.html#scenarios",
            "keywords": [*item.keywords, *item.api_names, *item.environment_ids],
        })
    for item in environment_catalog(language="en"):
        search_items.append({
            "kind": "environment",
            "lang": "en",
            "id": item.env_id,
            "title": item.title,
            "summary": item.summary,
            "path": "use-cases.html#envs",
            "keywords": [*item.keywords, *item.best_for],
        })
    for item in environment_catalog(language="zh"):
        search_items.append({
            "kind": "environment",
            "lang": "zh",
            "id": item.env_id,
            "title": item.title,
            "summary": item.summary,
            "path": "zh/use-cases.html#envs",
            "keywords": [*item.keywords, *item.best_for],
        })

    for route in common_eda_finding_routes():
        title_en, summary_en = route.title, route.summary
        title_zh, summary_zh = _route_title(route.route_id, route.title, route.summary, "zh")
        search_items.append({
            "kind": "eda_route",
            "lang": "en",
            "id": route.route_id,
            "title": title_en,
            "summary": summary_en,
            "path": "taskification.html#routing",
            "keywords": list(route.query_tokens) + list(route.recommended_tasks),
        })
        search_items.append({
            "kind": "eda_route",
            "lang": "zh",
            "id": route.route_id,
            "title": title_zh,
            "summary": summary_zh,
            "path": "zh/taskification.html#routing",
            "keywords": list(route.query_tokens) + list(route.recommended_tasks),
        })

    # Export tutorial notebooks and starter projects into the docs bundle.
    for asset in export_tutorial_notebooks(notebooks_dir, language="en"):
        result.notebook_files.extend([asset.path_ipynb, asset.path_py])
        search_items.append({
            "kind": "notebook",
            "lang": "en",
            "id": asset.notebook_id,
            "title": asset.title,
            "summary": asset.summary,
            "path": f"notebooks/{Path(asset.path_ipynb).name}",
            "keywords": [asset.tutorial_id, "notebook", "tutorial"],
        })
    for asset in export_tutorial_notebooks(zh_notebooks_dir, language="zh"):
        result.notebook_files.extend([asset.path_ipynb, asset.path_py])
        search_items.append({
            "kind": "notebook",
            "lang": "zh",
            "id": asset.notebook_id,
            "title": asset.title,
            "summary": asset.summary,
            "path": f"zh/notebooks/{Path(asset.path_ipynb).name}",
            "keywords": [asset.tutorial_id, "notebook", "tutorial"],
        })
    for item in starter_catalog(language="en"):
        project = create_starter_project(starters_dir / item.starter_id, item.starter_id, language="en")
        result.starter_projects.append(project.output_dir)
        search_items.append({
            "kind": "starter",
            "lang": "en",
            "id": item.starter_id,
            "title": item.title,
            "summary": item.summary,
            "path": f"starters/{item.starter_id}/README.md",
            "keywords": [*item.best_for, *item.api_names, *item.environment_ids],
        })
    for item in starter_catalog(language="zh"):
        project = create_starter_project(zh_starters_dir / item.starter_id, item.starter_id, language="zh")
        result.starter_projects.append(project.output_dir)
        search_items.append({
            "kind": "starter",
            "lang": "zh",
            "id": item.starter_id,
            "title": item.title,
            "summary": item.summary,
            "path": f"zh/starters/{item.starter_id}/README.md",
            "keywords": [*item.best_for, *item.api_names, *item.environment_ids],
        })
    from ..surface import demo as build_demo_bundle

    for spec in _showcase_specs("en"):
        bundle_dir = showcase_dir / spec["scenario"]
        bundle = build_demo_bundle(output_dir=bundle_dir, scenario=spec["scenario"])
        result.showcase_bundles.append(str(bundle.output_dir))
        search_items.append({
            "kind": "showcase",
            "lang": "en",
            "id": spec["scenario"],
            "title": spec["title"],
            "summary": spec["summary"],
            "path": f"showcase.html#{spec['section_id']}",
            "keywords": [spec["scenario"], "showcase", "report", "handoff"],
        })
    for spec in _showcase_specs("zh"):
        search_items.append({
            "kind": "showcase",
            "lang": "zh",
            "id": spec["scenario"],
            "title": spec["title"],
            "summary": spec["summary"],
            "path": f"zh/showcase.html#{spec['section_id']}",
            "keywords": [spec["scenario"], "showcase", "report", "handoff"],
        })
    for item in playbook_catalog(language="en"):
        search_items.append({
            "kind": "playbook",
            "lang": "en",
            "id": item.playbook_id,
            "title": item.title,
            "summary": item.summary,
            "path": f"playbooks.html#{item.playbook_id}",
            "keywords": [*item.api_names, *item.scenario_ids, *item.environment_ids],
        })
    for item in playbook_catalog(language="zh"):
        search_items.append({
            "kind": "playbook",
            "lang": "zh",
            "id": item.playbook_id,
            "title": item.title,
            "summary": item.summary,
            "path": f"zh/playbooks.html#{item.playbook_id}",
            "keywords": [*item.api_names, *item.scenario_ids, *item.environment_ids],
        })

    search_path = out / "search-index.json"
    search_path.write_text(json.dumps(search_items, ensure_ascii=False, indent=2), encoding="utf-8")
    result.search_index = str(search_path)

    api_manifest_path = out / "api-manifest.json"
    api_manifest_path.write_text(json.dumps(api_ref.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    result.api_manifest = str(api_manifest_path)

    route_map = []
    for route in common_eda_finding_routes():
        route_map.append({"lang": "en", **route.to_dict()})
        title_zh, summary_zh = _route_title(route.route_id, route.title, route.summary, "zh")
        payload = route.to_dict()
        payload["title"] = title_zh
        payload["summary"] = summary_zh
        route_map.append({"lang": "zh", **payload})
    route_map_path = out / "eda-route-map.json"
    route_map_path.write_text(json.dumps(route_map, ensure_ascii=False, indent=2), encoding="utf-8")
    result.eda_route_map = str(route_map_path)
    return result
