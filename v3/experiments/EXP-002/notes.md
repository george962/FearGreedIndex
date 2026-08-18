# EXP-002 — V3-006 Regularized Return Regression Baseline

This is the first v3 expected-return regression experiment, using `v3-features-001` and `v3-labels-001`.

## Method

- One Ridge regression per 5/20/60-session horizon.
- Median imputation and standard scaling are fit on training rows only.
- A row is eligible only after the relevant outcome-known date has matured by the fold cutoff.
- Fixed `alpha=1.0`; no hyperparameter search or post-hoc tuning.

## Observed result

The baseline is usable as a comparison point, but it materially underpredicted longer-horizon returns in multiple folds. The largest failure was 2025 at 60 sessions: mean actual return about +3.79% versus mean predicted return about -1.25%, with MAE about 9.93% and RMSE about 13.53%.

The exact 659-row prediction Parquet is preserved in GitHub Actions run `32197058166` (`v3-evidence-32197058166`) and its SHA-256 is recorded in `manifest.json`. Future experiments must compare against this record rather than overwrite it.
