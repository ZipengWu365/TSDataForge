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
