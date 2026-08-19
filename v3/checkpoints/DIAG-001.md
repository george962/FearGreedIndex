# DIAG-001 Checkpoint — Target, covariate, and relationship drift

## Purpose

DIAG-001 is diagnostic-only. It fits no model and does not select a champion. It analyzes all 53 registered Treasury-lane features on the exact EXP-006 realized test dates to explain the repeated 2024 -> 2025 -> 2026 prediction reversals.

## Main finding

**The dominant problem is broad conditional/relationship drift, accompanied by meaningful target and covariate drift. It is not explained by one static regime or by an overly long training window.**

### Target prevalence drift

The favorable-entry base rate itself moves materially relative to each fold's legally mature training history:

| Fold | Train prevalence | Test prevalence | Delta |
| --- | ---: | ---: | ---: |
| 2024 | 41.3% | 55.2% | **+13.8 pp** |
| 2025 | 45.9% | 41.6% | **-4.3 pp** |
| 2026 YTD | 45.3% | 35.0% | **-10.3 pp** |

### Feature-target relationship instability

Across all 53 features:

- **42/53 (79%)** have at least one training-to-test Spearman sign reversal.
- **34/53 (64%)** have at least one adjacent test-fold sign transition across 2024 -> 2025 -> 2026 YTD.
- The Fear & Greed family is especially unstable: **17/17** features reverse train-to-test sign at least once, and **16/17** change test sign across years.

Examples:

- `fear_greed`: training association is negative in every fold, but the test association flips positive in 2025 and negative again in 2026 YTD.
- `fg_change_5`: training/test signs disagree in **all three folds**.
- `interaction_fg_x_drawdown_20`: training/test signs disagree in **all three folds**.
- `treasury_10y_2y_slope`: sign changes across both training/test and calendar folds.

This explains why adding model complexity, static sentiment thresholds, or a single rate regime has not produced robust unseen performance.

### Covariate shift is also material

Large distribution shifts exist in several macro/market variables. Examples of maximum absolute standardized mean difference across folds include:

- `treasury_10y_level`: **1.45**
- `treasury_10y_percentile_252`: **1.42**
- `treasury_2y_level`: **1.06**
- `spx_distance_ma_200`: **1.01**
- `spx_drawdown_252` / `spx_distance_high_252`: about **0.98**
- `spx_realized_vol_60`: about **0.92**

Treasury and market features therefore occupy meaningfully different distributions across the research years.

## Stable-looking relationships worth hypothesis generation — not promotion evidence

A smaller group has **zero** training-to-test sign reversals and **zero** adjacent test-fold sign transitions. Notable examples:

| Feature | Direction | Test Spearman 2024 / 2025 / 2026 YTD | Raw upward AUC 2024 / 2025 / 2026 YTD |
| --- | --- | --- | --- |
| `spx_distance_ma_200` | negative | -0.410 / -0.330 / -0.431 | 0.262 / 0.307 / 0.239 |
| `spx_realized_vol_5` | positive | 0.143 / 0.119 / 0.132 | 0.583 / 0.569 / 0.580 |
| `spx_realized_vol_20` | positive | 0.190 / 0.138 / 0.449 | 0.610 / 0.581 / 0.772 |
| `spx_realized_vol_60` | positive | 0.216 / 0.269 / 0.405 | 0.625 / 0.658 / 0.745 |
| `treasury_slope_change_20` | positive | 0.056 / 0.163 / 0.220 | 0.532 / 0.596 / 0.633 |
| `treasury_10y_level` | positive | 0.016 / 0.150 / 0.286 | 0.509 / 0.588 / 0.673 |

These observations suggest that the more stable information may be related to **market stress / medium-term volatility / long-horizon market position**, not the raw Fear & Greed state itself.

However, `treasury_10y_level` and several other apparently stable relationships also have substantial covariate shift. Stability of sign alone is not enough.

## Critical research-governance consequence

DIAG-001 explicitly inspected the 2024, 2025, and 2026 YTD outcomes. Therefore any feature set, gating rule, or model formulation chosen because of DIAG-001 **must not treat those same folds as fresh promotion evidence**.

For post-DIAG research:

1. 2024–2026 YTD may be used as development/research evidence only.
2. Any adaptive feature-selection rule should be defined so that, inside each historical fold, it uses only information available before that fold.
3. Champion/promotion evidence must eventually include genuinely untouched data that was not used to formulate the post-DIAG hypothesis (for example a forward holdout beginning after the frozen 2026-08-18 research cutoff, or separately acquired historical data that has not been used in the research loop).
4. V3-019 remains blocked and sizing remains `1.00x`.

## Next step

Do **not** train another broad 53-feature model immediately.

The next development experiment should test a **past-only stability-selection procedure**: inside each outer fold, identify features whose association direction is stable across earlier chronological subperiods using training data only, then evaluate the frozen selection procedure on that fold. This can test whether stability filtering is a viable adaptive methodology, but because DIAG-001 already exposed the outer-fold outcomes, it remains development evidence rather than final promotion evidence.

In parallel, establish a forward untouched evaluation lane after the `2026-08-18` research cutoff so later candidates can accumulate genuinely unseen evidence.
