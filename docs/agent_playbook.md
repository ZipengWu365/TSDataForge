# TSDataForge Agent Playbook

TSDataForge is designed so agents do **not** begin with raw arrays.

## Recommended prompt skeleton

```text
Open handoff_index_min.json first.
Then follow agent_open_order.
Summarize dataset quality, main risks, and the single recommended next step.
Open action_plan.json only if more detail is required.
Do not open handoff_bundle.json unless a required field is missing.
```

## Why this works

- `dataset_context.json` is compact and semantic
- `dataset_card.md` is human-readable and stable
- `handoff_index_min.json` is the tiny routing contract
- `action_plan.json` makes next actions explicit

## Tool contracts

The `schemas/` directory now ships not only JSON Schemas for bundle artifacts, but also `tool_contracts.json` for the five public entry points:

- `load_asset`
- `report`
- `handoff`
- `taskify`
- `demo`
