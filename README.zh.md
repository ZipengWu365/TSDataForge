# TSDataForge

> **把原始时间序列文件变成报告、handoff bundle，以及 agent-ready 的下一步动作。**

<p align="left">
  <a href="https://zipengwu365.github.io/TSDataForge/"><img alt="Docs" src="https://img.shields.io/badge/docs-GitHub%20Pages-0b57d0"></a>
  <a href="https://github.com/ZipengWu365/TSDataForge"><img alt="GitHub" src="https://img.shields.io/badge/repo-TSDataForge-111827"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-16a34a"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-2563eb">
</p>

TSDataForge 是一个 **asset-first** 的时间序列 Python 库，也是一个 **time-series profiling and handoff layer**。

它最适合用在 **选模型之前** 的那个阶段：先把原始时间序列资产解释清楚，再决定后面进入 forecasting、anomaly、regime、causal 或 control。

给它一份原始时间序列资产，它会返回：

- 给人看的 **report**
- 用于交接的 **dataset card**
- 给 agent 用的 **compact context**
- 显式的 **decision record**
- 用于继续执行的 **next-action plan**

它**不是**主要做 forecasting 的模型工具包。
它**不是**为了替代 `sktime`、`Darts`、`tsfresh`、`STUMPY` 或 `YData Profiling`。
它关注的是 **dataset asset 本身**。

文档站点：
[zipengwu365.github.io/TSDataForge](https://zipengwu365.github.io/TSDataForge/)

## 什么时候适合用它

当你需要下面这些事情时，TSDataForge 比“直接挑模型”更合适：

- 先看清楚一份原始时间序列资产，再决定下游任务
- 把原始文件整理成可交接的 `report + bundle`
- 给同事、研究合作者或 agent 一个紧凑的上下文，而不是直接丢原始数组
- 在 forecasting、anomaly、regime、causal、control 等任务之间做更明确的路由

如果你现在只关心“训练哪个 estimator”或“比较哪个 forecasting benchmark”，那 TSDataForge 通常应该放在建模库之前，而不是替代建模库

## 三个真实公开数据 demo

- `python -m tsdataforge demo --scenario ecg_public --output ecg_bundle`
- `python -m tsdataforge demo --scenario macro_public --output macro_bundle`
- `python -m tsdataforge demo --scenario climate_public --output climate_bundle`

## 最短 happy path

```text
dataset -> report -> handoff bundle -> next action
```

## 安装

从 GitHub 直接安装：

```bash
pip install "git+https://github.com/ZipengWu365/TSDataForge.git"
```

如果你是本地克隆仓库，并且想启用可视化依赖：

```bash
pip install ".[viz]"
```

## 60 秒跑通一次

```python
from tsdataforge import demo

bundle = demo(output_dir="demo_bundle", scenario="ecg_public")
print(bundle.output_dir)
```

建议先打开 `demo_bundle/report.html`。

### 人类打开顺序

1. `report.html`
2. `dataset_card.md`
3. `dataset_context.json`
4. `handoff_index_min.json`
5. 需要更细时再看 `action_plan.json`

### Agent 打开顺序

1. `handoff_index_min.json`
2. 按 `agent_open_order` 继续打开
3. 需要更细时再看 `action_plan.json`
4. 执行 `recommended_next_step`

## 最该记住的五个 API

- `load_asset(...)`
- `report(...)`
- `handoff(...)`
- `taskify(...)`
- `demo(...)`

## 常用文档入口

- `docs/quickstart.md`：最快上手路径
- `docs/handoff.md`：handoff bundle 的核心定位
- `docs/showcase.md`：公开 demo 和展示素材
- `docs/api_reference.md`：API 总览
- `examples/README.md`：可直接运行的示例脚本

## 社区与引用

- [CONTRIBUTING.md](CONTRIBUTING.md)：贡献方式和提交流程
- [SECURITY.md](SECURITY.md)：安全问题私下报告方式
- [SUPPORT.md](SUPPORT.md)：提问或报 bug 时该带哪些信息
- [CITATION.cff](CITATION.cff)：引用元数据
- [LICENSE](LICENSE)：MIT 许可证

## 这个本地快照说明

这个产物已经包含源码、文档、schema、tool contracts 和 demo 资产。
真正发布到 GitHub Pages 或 PyPI，仍然需要你在外部 release workflow 里执行。
