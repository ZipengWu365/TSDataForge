# Public Demo Provenance

TSDataForge `0.3.7` promotes four built-in demos as **real public data**.

That claim should be easy to verify.

This page explains what each demo is built from and what has been transformed locally.

## What "real public demo" means here

- The underlying signal or table comes from a public sample dataset distributed by a well-known upstream library.
- TSDataForge repackages that source into local `.npz` assets under `tsdataforge/demo_data/`.
- The local assets are windowed or reshaped for report + handoff workflows.
- They are **not** the full upstream datasets, and they are **not** meant to replace the original sources.

## Demo provenance

| Scenario | Local file | Upstream source | What TSDataForge changes | Source docs |
|---|---|---|---|---|
| `ecg_public` | `tsdataforge/demo_data/ecg_public.npz` | SciPy `electrocardiogram()` example data, based on an excerpt from MIT-BIH Arrhythmia Database record 208 | slices the public ECG signal into fixed windows and wraps them as a `SeriesDataset` | [SciPy electrocardiogram docs](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.datasets.electrocardiogram.html) |
| `macro_public` | `tsdataforge/demo_data/macro_public.npz` | `statsmodels` `macrodata` sample, a US macroeconomic panel | selects and windows macro indicators for handoff-first workflows | [statsmodels macrodata docs](https://www.statsmodels.org/dev/datasets/generated/macrodata.html) |
| `climate_public` | `tsdataforge/demo_data/climate_co2_public.npz` | `statsmodels` `co2` sample, the classic Mauna Loa weekly atmospheric CO2 series | windows the weekly series and packages it as a reusable handoff asset | [statsmodels CO2 docs](https://www.statsmodels.org/devel/datasets/generated/co2) |
| `sunspots_public` | `tsdataforge/demo_data/sunspots_public.npz` | `statsmodels` `sunspots` sample, yearly sunspot counts | windows the yearly sequence for long-cycle handoff demos | [statsmodels sunspots docs](https://www.statsmodels.org/stable/datasets/generated/sunspots.html) |

## Practical trust notes

- `ecg_public` is the strongest medicine-facing public demo because the upstream source is explicit and recognizable.
- `macro_public`, `climate_public`, and `sunspots_public` are convenient public examples from `statsmodels`; they are useful for onboarding and demos, but they should still be described as **sample public datasets**, not as exhaustive production datasets.
- If you showcase these demos on GitHub, link both this page and the upstream source page so users can verify the claim quickly.

## Recommended README phrasing

Use wording like:

`Built from public sample datasets bundled by SciPy or statsmodels; see docs/public_data_provenance.md for source notes.`

Avoid wording like:

`Real public data` without any visible source link nearby.
