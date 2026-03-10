# TSDataForge Handoff

The central product idea is:

```text
dataset -> report -> handoff bundle -> next action
```

## What the bundle is for

A handoff bundle turns one time-series asset into a small, shareable, predictable directory containing:

- `report.html`
- `dataset_card.md`
- `dataset_context.json`
- `handoff_index_min.json`
- `handoff_index.json`
- `action_plan.json`
- `handoff_bundle.json`
- `schemas/`

## Agent open order

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

This keeps the first agent read small, while preserving a richer execution plan when needed.
