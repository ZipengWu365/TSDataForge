# TSDataForge Handoff

The central product idea is:

```text
dataset -> report -> handoff bundle -> next action
```

Use this path when you need to understand a raw time-series dataset before modeling or before handing it to another person, script, or agent.

Open `report.html` first. Everything else in the folder exists to explain the dataset or make the next step explicit.

## What the bundle is for

A handoff bundle turns one time-series dataset into a small, shareable, predictable directory containing:

- `report.html`
- `dataset_card.md`
- `dataset_context.json`
- `handoff_index_min.json`
- `handoff_index.json`
- `action_plan.json`
- `handoff_bundle.json`
- `schemas/`

## Use your own data with `report(...)` and `handoff(...)`

### One saved file

```python
from tsdataforge import handoff, report

report(
    "my_dataset.npy",
    output_path="report.html",
    dataset_id="factory_sensor_run",
)

bundle = handoff(
    "my_dataset.npy",
    output_dir="dataset_handoff_bundle",
    dataset_id="factory_sensor_run",
)
```

### Arrays plus explicit time and channel names

```python
import pandas as pd
from tsdataforge import handoff, report

df = pd.read_csv("pump_run.csv")
values = df[["temperature", "pressure"]].to_numpy()
time = df["seconds"].to_numpy()

report(
    values,
    time=time,
    output_path="report.html",
    dataset_id="pump_run",
    channel_names=["temperature", "pressure"],
)

bundle = handoff(
    values,
    time=time,
    output_dir="pump_bundle",
    dataset_id="pump_run",
    channel_names=["temperature", "pressure"],
)
```

### Input rules

- direct `.csv` / `.txt` loading expects numeric files
- if the first column is monotonic increasing, it is used as `time`
- `values.shape == (n_series, length)` means one row per series
- `values.shape == (length, n_channels)` plus `time.shape == (length,)` means one multichannel series

## Automation / agent open order

1. open `handoff_index_min.json`
2. follow `agent_open_order`
3. open `action_plan.json` only if more detail is needed
4. execute `recommended_next_step`

## Human open order

1. open `report.html`
2. open `dataset_card.md`
3. open `dataset_context.json`
4. open `handoff_index_min.json`

## Why there are two index layers

- `handoff_index_min.json` is the **smallest first-entry contract**
- `handoff_index.json` is a slightly richer routing map
- `action_plan.json` contains the detailed already_done / recommended / optional breakdown

This keeps the first automation or agent read small, while preserving a richer execution plan when needed.

## Install note

TSDataForge is not on PyPI yet. Install from GitHub with:

```bash
pip install "git+https://github.com/ZipengWu365/TSDataForge.git"
```
