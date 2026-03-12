# TSDataForge Examples

These examples are intentionally short and scenario-first.

## Fastest real public demos

- `real_public_ecg_handoff.py`
- `real_public_macro_handoff.py`
- `real_public_climate_handoff.py`

## Common handoff patterns

- `real_csv_to_report_30s.py`
- `profile_your_own_csv.py`
- `agent_only_open_order.py`
- `compare_two_dataset_versions.py`

## Bring-your-own-data note

- `profile_your_own_csv.py` is the human-first example: read your table, pick columns, call `report(...)`, then optionally `handoff(...)`
- `real_csv_to_report_30s.py` shows the direct numeric-CSV path
- if your source file has headers or timestamps, load it yourself and pass arrays into `load_asset(...)`, `report(...)`, or `handoff(...)`

## Why these examples exist

They are designed to answer three practical questions quickly:

1. how do I turn a real asset into a report?
2. how do I generate a handoff bundle another person or agent can use?
3. how do I move from a base dataset to a task-specific dataset only after the asset is understood?
