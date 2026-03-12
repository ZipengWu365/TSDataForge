# TSDataForge API Reference Guide

Start from the **public surface**:

- `load_asset(...)`
- `report(...)`
- `handoff(...)`
- `taskify(...)`
- `demo(...)`

Then move to the advanced surface only when needed.

## Public-surface rule of thumb

- use `report(...)` when a human needs the first explanation
- use `handoff(...)` when the asset must travel to another person or agent
- use `taskify(...)` only after the asset is understood
- use `demo(...)` for GitHub, docs, workshops, and smoke tests

## `load_asset(...)` input rules

- direct file loading supports `.npy`, `.npz`, `.csv`, `.txt`, and `.json`
- direct `.csv` / `.txt` loading expects numeric files
- if the first column is monotonic increasing, it is interpreted as `time`
- `channel_names=` is useful for multichannel arrays and handoff bundles
- if your source is a pandas DataFrame, pass `df.to_numpy()` and an explicit `time=` array
