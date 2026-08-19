# EXP-006 — Opportunity-State Target Reformulation

Status: **PRE-REGISTERED / ACTIVE**  
Issue: #46  
Research cutoff: **2026-08-18**

## Question

Can the existing point-in-time Fear/Greed + Treasury feature set predict a directly actionable **20-session entry opportunity** more reliably than the earlier exact-return/drawdown formulation?

EXP-006 deliberately changes the **target formulation**, not the production policy, sizing, or champion gates.

## Frozen inputs

- Feature set: `v3-features-004-treasury` (53 model features)
- Existing executable-entry labels: `forward_return_20d`, `max_drawdown_20d`, `_forward_20d_known_date`
- Entry convention: signal on T, entry next tradable session open
- Folds: test 2024, 2025, 2026 YTD; training cutoffs unchanged from `config.json`
- Final research as-of cutoff: 2026-08-18

## Pre-registered targets

### Primary binary target

`favorable_entry_20d = 1` only when both are true:

- `forward_return_20d >= +0.02`
- `max_drawdown_20d > -0.05`

Otherwise the mature observation is `0`.

### Diagnostic ordinal target

`opportunity_state_20d` uses this precedence:

1. `BAD` if `forward_return_20d <= -0.02` OR `max_drawdown_20d <= -0.05`
2. `EXCELLENT` if `forward_return_20d >= +0.05` and drawdown is above -5%
3. `GOOD` if return is at least +2% but below +5% and drawdown is above -5%
4. `NORMAL` otherwise

The ordinal state is diagnostic in EXP-006; it is **not** used to tune thresholds or select a model.

## Fixed models

Two intentionally simple baselines run on identical samples:

1. `opportunity_logistic_l2_v1`
   - median imputation
   - standardization
   - L2 logistic regression
   - `C=1.0`, `lbfgs`, `max_iter=1000`

2. `opportunity_random_forest_v1`
   - median imputation
   - 300 trees
   - max depth 6
   - min leaf 10
   - sqrt feature sampling
   - bootstrap true
   - seed 42
   - one thread

No hyperparameter search is permitted under EXP-006.

## Point-in-time training rule

A row may enter training only when:

- its decision date is on/before the fold train cutoff,
- `_forward_20d_known_date` is on/before the fold train cutoff,
- the opportunity target is non-missing.

A test row is scored only when the 20-session outcome is known on/before 2026-08-18.

## Primary viability gate

EXP-006 passes only if **at least one** of the two fixed models satisfies every condition:

- mean relative Brier improvement versus the fold-specific **training-prevalence** base-rate predictor > 0,
- positive relative-Brier improvement in at least 2 of 3 chronological folds,
- mean ROC AUC > 0.52,
- ROC AUC > 0.50 in at least 2 of 3 folds,
- test-sample hashes are identical across the two models for every fold.

ECE, log loss, precision, recall, class prevalence, and ordinal-state distributions are diagnostics only.

## Decision after the run

### If PASS

Freeze EXP-006 as positive evidence that opportunity-state prediction is learnable. Start a **new** experiment ID for explicit Treasury regime conditioning. Do not tune EXP-006.

### If FAIL

Freeze EXP-006 as a negative target-formulation experiment. Do not change thresholds or model parameters under EXP-006. The next experiment must change the formulation in a pre-registered way.

## Outputs

Compact committed evidence:

- `v3/reports/exp006_evaluation.json`
- `v3/reports/exp006_metrics.csv`
- `v3/reports/exp006_state_distribution.csv`
- `v3/experiments/EXP-006/manifest.json`
- final checkpoint under `v3/checkpoints/EXP-006.md`

Generated prediction Parquet remains CI/rebuild output and is not committed.
