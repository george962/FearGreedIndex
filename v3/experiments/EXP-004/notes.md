# EXP-004 — V3-008 Random-Forest Benchmark

This experiment is the fixed nonlinear robustness benchmark using `v3-features-001` and `v3-labels-001` under the same chronological folds as EXP-001 through EXP-003.

## Method

- Random-forest classification for positive 5/20/60-session returns.
- Random-forest regression for expected 5/20/60-session returns and 20-session maximum drawdown.
- Fixed 300 trees, max depth 6, minimum 10 samples per leaf, square-root feature sampling, bootstrap enabled, seed 42, single-process fitting.
- Median imputation is fit on training rows only.
- Every target uses the shared outcome-known-date maturity gate.
- No hyperparameter search or post-hoc tuning.

## Observed result

Random forest has the strongest aggregate predictive metrics of the four experiments so far: mean classification Brier about 0.232, mean classification log loss about 0.662, mean return MAE about 3.25%, and mean return RMSE about 4.12%. It is still not promoted here because V3-008 is explicitly a robustness benchmark and formal model comparison belongs to V3-009/V3-010.

The exact 659-row prediction Parquet is preserved in GitHub Actions run `32197829836` (`v3-evidence-32197829836`) and its SHA-256 is recorded in `manifest.json`.
