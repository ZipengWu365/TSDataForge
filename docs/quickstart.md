# TSDataForge Quickstart

TSDataForge is a **time-series profiling and handoff layer**.

Start here if you want the fastest path from a raw dataset to a profiling report first, then a shareable handoff folder only when you need it.

## Install

From GitHub:

```bash
pip install "git+https://github.com/ZipengWu365/TSDataForge.git"
```

From a local clone with report plots and the GUI:

```bash
pip install ".[viz]"
```

PyPI publishing is not live yet, so `pip install tsdataforge` will not work until a package release is published there.

## 30-second path

```python
from tsdataforge import demo

bundle = demo(output_dir="demo_bundle", scenario="ecg_public")
print(bundle.output_dir)
```

Default demo in `0.3.7` is a **real public ECG** showcase.

Then open:

1. `demo_bundle/report.html`
2. `demo_bundle/dataset_card.md`
3. `demo_bundle/dataset_context.json`
4. `demo_bundle/handoff_index_min.json`

## Local GUI path

If you do not want to start from a terminal workflow, launch the local GUI:

```bash
git clone https://github.com/ZipengWu365/TSDataForge.git
cd TSDataForge
pip install ".[viz]"
python -m tsdataforge ui
```

Then open `http://127.0.0.1:8765/`, drag in a raw file, and inspect the generated:

1. `report.html`
2. `dataset_card.md`
3. `dataset_context.json`
4. `handoff_index_min.json`

## Use your own data

### One saved file

If you already have a saved dataset file, start here:

```python
from tsdataforge import report

report(
    "my_dataset.npy",
    output_path="report.html",
    dataset_id="lab_measurements",
)
```

Open `report.html` first. If you need the shareable folder afterwards, call `handoff(...)` on the same file.

Supported direct inputs:

- `.npy`
- `.npz`
- `.csv`
- `.txt`
- `.json`

### Arrays or pandas DataFrame

If your CSV has headers, timestamps, or explicit channel names, convert it to arrays first:

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

### Shape rules

- `values.shape == (length,)`: one univariate series
- `values.shape == (n_series, length)`: one row per series
- `values.shape == (length, n_channels)` with `time.shape == (length,)`: one multichannel series
- `values.shape == (n_series, length, n_channels)`: many multichannel series

### CSV rule

- Direct `.csv` and `.txt` loading expects numeric files
- if the first column is monotonic increasing, it is treated as `time`
- if your file has headers or date strings, load it yourself and pass `values` plus `time=`

## 5-minute path

```python
from tsdataforge import demo

bundle = demo(output_dir="macro_bundle", scenario="macro_public")
print(bundle.output_dir)
```

Read the report first. Then inspect the card and the minimal handoff index.

## 20-minute path

```python
from tsdataforge import handoff, taskify

bundle = handoff("my_dataset.npy", output_dir="my_bundle")
forecast = taskify("my_dataset.npy", task="forecasting", horizon=24)
```

Only call `taskify(...)` **after** the report makes sense.
