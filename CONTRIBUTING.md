# Contributing

Thanks for contributing to TSDataForge.

The project is still in an alpha stage, so the best contributions are the ones that make the public story clearer, safer, and easier to trust.

## Good contribution targets

- bug fixes in `report(...)`, `handoff(...)`, `taskify(...)`, or `demo(...)`
- loader improvements for real `.npy`, `.npz`, `.csv`, `.txt`, or `.json` assets
- docs and examples that make the asset-first workflow easier to understand
- tests that protect handoff artifacts, schema outputs, and public demos
- release hygiene fixes for docs, showcase assets, and generated bundle expectations

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,viz]"
```

Run the test suite before opening a pull request:

```bash
pytest
```

## Development guidance

- Keep the core product identity stable: TSDataForge is a time-series profiling and handoff layer, not primarily a model-training framework.
- Prefer improving the first-success path over adding surface area. The five APIs users should remember are `load_asset`, `report`, `handoff`, `taskify`, and `demo`.
- If you change public behavior, update the matching README, docs page, and example script in the same pull request.
- Preserve generated artifact names and bundle structure when possible. If a change is intentionally breaking, call it out clearly in the pull request.
- Keep dependencies lightweight unless there is a strong release-quality reason to add one.

## Pull request checklist

Before opening a pull request, please make sure you have:

1. added or updated tests for the behavior you changed
2. run `pytest`
3. updated README, docs, examples, or showcase assets if the public surface changed
4. explained the motivation and user-facing impact in the pull request description
5. included before/after output, screenshots, or generated artifact examples if the change affects reports or handoff bundles

## Reporting problems

- Usage questions: open a GitHub issue and include a minimal reproducible example.
- Regressions: include the version or commit, input shape, and the generated artifact that shows the problem.
- Security issues: do not open a public issue first; follow [SECURITY.md](SECURITY.md).
