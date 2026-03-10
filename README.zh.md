# TSDataForge

> **把原始时间序列文件变成报告、handoff bundle，以及 agent-ready 的下一步动作。**

TSDataForge 是一个 **asset-first** 的时间序列 Python 库。

它最适合用在 **选模型之前** 的那个阶段：先把原始时间序列资产解释清楚，再决定后面进入 forecasting、anomaly、regime、causal 或 control。

给它一份原始时间序列资产，它会返回：

- 给人看的 **report**
- 用于交接的 **dataset card**
- 给 agent 用的 **compact context**
- 用于继续执行的 **next-action plan**

它**不是**主要做 forecasting 的模型工具包。
它**不是**为了替代 `sktime`、`Darts`、`tsfresh`、`STUMPY` 或 `YData Profiling`。
它关注的是 **dataset asset 本身**。

## 三个真实公开数据 demo

- `python -m tsdataforge demo --scenario ecg_public --output ecg_bundle`
- `python -m tsdataforge demo --scenario macro_public --output macro_bundle`
- `python -m tsdataforge demo --scenario climate_public --output climate_bundle`

## 最短 happy path

```text
dataset -> report -> handoff bundle -> next action
```

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

## 这个本地快照说明

这个产物已经包含源码、文档、schema、tool contracts 和 demo 资产。
真正发布到 GitHub Pages 或 PyPI，仍然需要你在外部 release workflow 里执行。
