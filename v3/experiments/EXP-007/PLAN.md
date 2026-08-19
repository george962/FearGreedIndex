# EXP-007 — 20-Session 10Y Rate-Regime Test

Status: **PRE-REGISTERED / ACTIVE**  
Issue: #48  
Research cutoff: **2026-08-18**

## Question

EXP-006 showed severe chronological instability for the fixed favorable-entry target. Does a simple, point-in-time **10Y rate-trend regime** itself carry stable information about 20-session entry opportunity?

This experiment intentionally uses **no ML model**. It tests the regime hypothesis before allowing regime-specific model complexity.

## Frozen inputs

- Feature set: `v3-features-004-treasury` (53 features)
- Primary target: exact EXP-006 `favorable_entry_20d`
- Favorable = `forward_return_20d >= +2%` AND `max_drawdown_20d > -5%`
- Labels mature only when `_forward_20d_known_date` is available by the training cutoff
- Folds: 2024 / 2025 / 2026 YTD from `config.json`
- Research as-of: 2026-08-18

## Single frozen regime

Use the existing point-in-time feature `treasury_10y_change_20` only:

- `RATES_RISING` if `treasury_10y_change_20 > 0`
- `RATES_FALLING_OR_FLAT` if `treasury_10y_change_20 <= 0`

No alternate rate horizon, threshold, or Treasury feature is tested under EXP-007.

## Predictor

For each fold:

1. Build the legally mature historical training sample.
2. Compute the global favorable-entry training prevalence.
3. Compute favorable-entry prevalence separately for each fixed rate regime.
4. For each test date, predict its favorable-entry probability using the historical prevalence of its decision-date regime.

The reference prediction is the global training prevalence for every test row.

Each regime must have at least **100 eligible training rows** and both target classes in every fold. Every mature test row must have a non-missing regime.

## Frozen viability gate

EXP-007 passes only if all are true:

- 100% mature test-row regime coverage,
- exact realized-date sample hashes match EXP-006,
- every fold has >=100 training rows and both classes in each regime,
- mean relative Brier improvement versus global training prevalence > 0,
- positive relative-Brier improvement in at least 2/3 folds,
- mean ROC AUC > 0.52,
- ROC AUC > 0.50 in at least 2/3 folds,
- minimum fold ROC AUC >= 0.45.

The final condition prevents a superficially positive average from hiding another severe sign reversal.

## Diagnostics

Preserve:

- global and per-regime training prevalence,
- per-regime training counts,
- per-regime test counts and realized favorable rates,
- Brier / baseline Brier / relative Brier improvement,
- log loss / ECE / ROC AUC,
- training-prevalence ordering between the two regimes in every fold.

## Decision

### If PASS

Freeze EXP-007 as positive evidence that the rate-trend regime is informative. A **new** experiment may then test regime-specific models.

### If FAIL

Freeze EXP-007 as a negative regime hypothesis. Do not try alternate Treasury thresholds/features under EXP-007. Any next regime hypothesis gets a new experiment ID.

EXP-007 never promotes a champion, unlocks V3-019, or changes 1.00x sizing.
