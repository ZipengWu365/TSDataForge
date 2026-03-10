# TSDataForge Token Benchmark

The breakout story is not only that TSDataForge makes nice reports.
It is that it creates **smaller, more stable artifacts** for agent and teammate handoff.

## The benchmark question

How much smaller is the semantic handoff surface than a raw array dump?

## Typical pattern

- raw JSON from arrays: very large
- `dataset_context.json`: compact semantic layer
- `dataset_card.md`: compact human handoff layer
- `handoff_index_min.json`: smallest first-entry routing contract
- `action_plan.json`: richer execution detail only when needed

The exact token numbers vary by dataset, but the product direction is stable: **understand the dataset semantically before reopening the raw asset**.
