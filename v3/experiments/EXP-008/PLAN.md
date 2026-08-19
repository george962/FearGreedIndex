# EXP-008 — Fear & Greed extreme-state hypothesis

## Question

Does a fixed decision-date Fear & Greed state partition carry stable information about the 20-session favorable-entry target that EXP-006 could not learn stationarily?

## Frozen inputs

- research cutoff: `2026-08-18`
- feature set: `v3-features-004-treasury`
- target source: EXP-006 `favorable_entry_20d`
- same executable next-session entry convention
- same chronological folds: 2024 / 2025 / 2026 YTD
- same outcome-known-date maturity rule
- same realized test-date hashes as EXP-006

## Frozen state definition

Using decision-date `fear_greed` only:

- `EXTREME_FEAR`: `fear_greed <= 25`
- `NEUTRAL_RANGE`: `25 < fear_greed < 75`
- `EXTREME_GREED`: `fear_greed >= 75`

No threshold search or percentile substitution is allowed under EXP-008.

## Predictor

No ML model. Each test row receives the legally mature historical favorable-entry prevalence of its current sentiment state. The reference predictor is the global prevalence from the same eligible training sample.

Each state must have at least 50 training rows and both target classes in every fold. Missing state on a mature test row is a hard failure.

## Viability gate

EXP-008 passes only if all are true:

1. full mature-test state coverage;
2. exact EXP-006 sample hashes;
3. mean relative Brier improvement > 0;
4. positive relative Brier in at least 2/3 folds;
5. mean ROC AUC > 0.52;
6. ROC AUC > 0.50 in at least 2/3 folds;
7. minimum fold ROC AUC >= 0.45;
8. training prevalence order `EXTREME_FEAR > NEUTRAL_RANGE > EXTREME_GREED` in at least 2/3 folds.

## Interpretation

A PASS only justifies a new experiment for an extreme-state-only or gated model. It does not select a champion, unlock V3-019, or alter sizing.

A FAIL freezes this exact 25/75 hypothesis as negative. Thresholds may not be changed under the EXP-008 ID after results are seen.
