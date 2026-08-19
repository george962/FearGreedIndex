# STAB-001 — Past-only relationship stability selection

## Question

Can the system improve robustness by using only features whose relationship with the 20-session favorable-entry target has remained directionally consistent across multiple **earlier** subperiods, instead of fitting all 53 features as if market relationships were stationary?

## Why this is next

DIAG-001 found broad feature-target drift: most features changed direction across exposed years, and the Fear & Greed family was especially unstable. EXP-007 through EXP-009 also rejected a simple fixed rate regime, fixed sentiment extremes, and a fixed recent training window.

STAB-001 therefore changes the feature-selection methodology before another predictive model is trained.

## Frozen inputs

- research cutoff: `2026-08-18`
- feature set: `v3-features-004-treasury` (53 features)
- target: EXP-006 `favorable_entry_20d`
- exact EXP-006 2024 / 2025 / 2026-YTD outer test samples
- same executable next-session entry convention
- same outcome-known-date maturity rule
- EVID-001 outcomes remain sealed

The outer folds are development evidence only because DIAG-001 already inspected them.

## Frozen selection rule

For each outer fold independently:

1. Form the legal training set using only rows whose 20-session outcome was known by that fold's `train_end`.
2. Sort the legal training set chronologically and split it into four contiguous, as-equal-as-possible blocks.
3. For every registered feature, compute Spearman association with `favorable_entry_20d` separately in each block.
4. A feature qualifies only when:
   - every block has at least 100 observed rows;
   - at least 3/4 block associations have the same non-zero sign;
   - the most recent block agrees with that majority sign;
   - median absolute block Spearman is at least `0.05`.
5. Direction is the majority sign.
6. Weight is `median_abs_spearman * sign_consistency_fraction`.
7. No feature may be added because of outer-test performance or because DIAG-001 highlighted it.

## Frozen stability score

Within each outer fold:

- convert each selected feature to an empirical percentile using the legal training distribution only;
- center each percentile at `0.5`;
- multiply by learned direction and frozen stability weight;
- sum and normalize by total absolute weight;
- transform the composite score to `[0,1]` using the legal training-score empirical CDF only.

If no feature qualifies, use the legal training base rate as the whole-fold probability and fail the feature-support condition.

This deliberately tests the value of stable relationships **before** introducing another ML algorithm.

## Feature families

For diversity checks:

- `fear_greed`: `fear_greed` and `fg_*`
- `treasury`: `treasury_*`
- `spx_interaction`: SPX and interaction features

## Viability gate

STAB-001 passes only if all are true:

1. at least 3 features are selected in at least 2/3 outer folds;
2. at least 2 feature families are represented in at least 2/3 folds;
3. mean ROC AUC > `0.52`;
4. ROC AUC > `0.50` in at least 2/3 folds;
5. minimum fold AUC >= `0.45`;
6. mean relative Brier improvement versus each fold's legal training base-rate predictor > `0`;
7. positive relative Brier improvement in at least 2/3 folds;
8. all outer test sample hashes match EXP-006 exactly.

## Interpretation

**PASS:** create a separately pre-registered `EXP-010` adaptive predictive model using only the past-only selector. EXP-010 must separately freeze any recency weighting, model class, regime inputs, and calibration.

**FAIL:** do not fit EXP-010 from this selector. Preserve the failure and revisit data/target representation under a new methodology or experiment ID.

STAB-001 cannot select a champion, unlock V3-019, change the decision policy, or change sizing.
