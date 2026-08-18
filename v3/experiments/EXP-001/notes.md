# EXP-001 — V3-005 Logistic Classification Baseline

This is the first frozen v3 model experiment. It uses the 41-feature `v3-features-001` feature set and `v3-labels-001` executable-entry labels.

## Method

- One L2-regularized logistic classifier per 5/20/60-session horizon.
- Median imputation and standard scaling are fit on training rows only.
- A training row is eligible only after that horizon's outcome-known date is at or before the fold cutoff.
- Fixed `C=1.0`, `liblinear`, and random seed 42.
- No hyperparameter search and no post-hoc threshold tuning.

## Observed result

The model is a valid chronological baseline, not a champion. Several folds are weak or overconfident. The most obvious failure is the 2025 60-session model (Brier about 0.478, log loss about 1.995). These results are intentionally preserved rather than retuned after observation.

The exact 659-row `predictions.parquet` used for these metrics is preserved in GitHub Actions run `32196286533` (`v3-evidence-32196286533`) under `v3/reports/logistic_baseline_predictions.parquet`. Its SHA-256 is recorded in `manifest.json` as `389de8db830e2b9771eb5ea72fbd0b2b4ec52e115b318944513256d1ad769de7`.

Future V3-006 through V3-010 work must compare against this recorded experiment using the common evaluator rather than rewriting EXP-001.
