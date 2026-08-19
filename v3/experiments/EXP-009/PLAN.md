# EXP-009 — Fixed 504-observation recent-window adaptation

## Question

Does limiting training to the most recent two trading years of legally mature observations reduce the chronological instability seen in EXP-006 through EXP-008?

## Frozen inputs

- research cutoff: `2026-08-18`
- feature set: `v3-features-004-treasury` (53 features)
- target: EXP-006 `favorable_entry_20d`
- same executable next-session entry convention
- same 2024 / 2025 / 2026 YTD test folds
- same outcome-known-date maturity rule
- exact EXP-006 realized test-date hashes

## Frozen adaptation rule

For each fold, create the legally eligible training sample first, sort chronologically, then keep exactly the **most recent 504 eligible rows**.

No alternate window length, weighting scheme, regime filter, threshold search, or hyperparameter change is allowed under EXP-009.

## Frozen model

Reuse the EXP-006 random forest exactly:

- median imputation
- 300 trees
- max depth 6
- minimum leaf size 10
- `sqrt` max features
- bootstrap enabled
- random seed 42
- single-threaded fit
- no calibration

## Comparisons

1. recent-window fold base rate;
2. frozen full-history EXP-006 random forest on the exact same test dates.

## Viability gate

EXP-009 passes only if all are true:

1. exactly 504 training rows in every fold;
2. exact EXP-006 test-date hashes;
3. mean relative Brier improvement vs recent-window base rate > 0;
4. positive relative Brier in at least 2/3 folds;
5. mean ROC AUC > 0.52;
6. AUC > 0.50 in at least 2/3 folds;
7. minimum fold AUC >= 0.45;
8. mean Brier better than full-history EXP-006 RF;
9. mean AUC better than full-history EXP-006 RF;
10. Brier better than full-history RF in at least 2/3 folds;
11. AUC better than full-history RF in at least 2/3 folds.

A PASS justifies a new experiment for adaptive prediction/calibration. It does not select a champion or change sizing.

A FAIL freezes the 504-row hypothesis. Another recency timescale requires a new experiment ID.
