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

Future V3-006 through V3-010 work must compare against this recorded experiment using the common evaluator rather than rewriting EXP-001.
