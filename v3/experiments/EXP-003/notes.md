# EXP-003 — V3-007 Gradient-Boosting Candidate

This is the first fixed nonlinear v3 candidate, using `v3-features-001` and `v3-labels-001` under the same chronological folds as EXP-001 and EXP-002.

## Method

- Histogram gradient boosting classification for positive 5/20/60-session returns.
- Histogram gradient boosting regression for expected 5/20/60-session returns and 20-session maximum drawdown.
- Fixed learning rate 0.05, 150 boosting iterations, 15 max leaf nodes, 20 minimum samples per leaf, L2 regularization 1.0, seed 42.
- Median imputation is fitted on training rows only.
- Every target uses its own outcome-known-date maturity gate.
- No hyperparameter search or post-hoc tuning.

## Observed result

The candidate is mixed, not a clear champion. It performs strongly in some long-horizon 2026 YTD cases, including 60-session classification (Brier about 0.107, log loss about 0.422) and 60-session return prediction (MAE about 3.73%, RMSE about 4.48%). However, several 2024 and 2025 probability folds are substantially worse than the simpler logistic baseline, including 2024 60-session Brier about 0.379 and 2025 20-session Brier about 0.442.

The exact 659-row prediction Parquet is preserved in GitHub Actions run `32197501410` (`v3-evidence-32197501410`) and its SHA-256 is recorded in `manifest.json`. The mixed result is preserved unchanged for the common V3-009/V3-010 evaluation.
