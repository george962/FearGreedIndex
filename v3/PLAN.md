# FearGreedIndex v3 Development Plan

## Goal

Build a genuinely predictive, multi-factor tactical-allocation research system that can outperform the frozen v2.1 baseline on unseen chronological data without relying on retrospective threshold tuning.

v3 must remain research-only until it passes the acceptance gates below. The current v2.1 system remains the operational benchmark and continues collecting its live signal ledger.

---

## Development rules

1. **Never tune on the final test period.** All model selection and feature decisions must use chronological training/validation only.
2. **One change family at a time.** Add features or model classes incrementally so we can measure what actually improves results.
3. **Same evaluator for every candidate.** All models use the same folds, labels, transaction-cost assumptions, and portfolio rules.
4. **No dashboard-driven research logic.** Research produces versioned outputs; the dashboard displays approved outputs.
5. **No tactical sizing optimization until predictive edge is demonstrated.** Start with baseline exposure and one simple positive-overlay tier.
6. **Every experiment gets an ID and immutable result summary.** Do not overwrite prior experiment evidence.
7. **v2.1 stays frozen.** Any material methodology change belongs to v3.

---

# Phase 0 — Preserve the benchmark

## V3-001 Freeze v2.1 baseline

### Work
- Preserve the v2.1 strategy version and configuration.
- Generate a permanent baseline result package.
- Record the exact dataset coverage and evaluation assumptions.
- Preserve the current signal ledger.

### Required outputs
- `reports/baseline_v2_1/backtest_summary.json`
- `reports/baseline_v2_1/walk_forward_summary.csv`
- `reports/baseline_v2_1/action_scorecard.csv`
- `reports/baseline_v2_1/README.md`

### Acceptance criteria
- Baseline can be reproduced from a clean checkout.
- Strategy version/config hashes are recorded.
- Tactical sizing remains disabled.
- Later v3 reports can compare directly against these results.

---

# Phase 1 — Build the research dataset

## V3-002 Create the point-in-time feature dataset

### Initial Fear & Greed features
- level
- 1/3/5/10-observation changes
- rolling 5/20-day min and max
- distance from recent sentiment extremes
- 60-day and 252-day percentile/rank
- sentiment acceleration/reversal features

### Initial market features
- SPX 1/3/5/10/20/60-day returns
- distance from 20/50/200-day moving averages
- distance from 20/60/252-day highs
- 5/20/60-day realized volatility
- recent drawdown and rebound measures

### Initial interactions
- fear/greed × drawdown
- fear/greed × volatility
- sentiment change × market return

### Required outputs
- `v3/features/build_features.py`
- `v3/features/feature_registry.json`
- `v3/data/features_daily.parquet`

### Acceptance criteria
- One row per decision date.
- Every feature uses information available on or before that date.
- Deterministic generation.
- No duplicated dates.
- Missing-data statistics are produced.

---

## V3-003 Add leakage and data-quality validation

### Tests
- no future market observations used in features
- no forward returns in feature columns
- deterministic rebuild produces identical data/hash
- correct next-session entry alignment
- unique/sorted dates
- feature ranges and missingness checks
- training folds cannot access later observations

### Required outputs
- `v3/tests/test_features.py`
- `v3/tests/test_leakage.py`
- `v3/evaluation/validate_dataset.py`

### Acceptance criteria
- CI fails if leakage is introduced.
- Dataset validation passes before model training can run.

---

# Phase 2 — Build explicit prediction targets

## V3-004 Create multi-horizon labels

### Return labels
- `forward_return_5d`
- `forward_return_20d`
- `forward_return_60d`

### Classification labels
- `forward_positive_5d`
- `forward_positive_20d`
- `forward_positive_60d`

### Risk labels
- `max_drawdown_5d`
- `max_drawdown_20d`
- `max_drawdown_60d`
- probability/indicator of an additional 5% decline over the next 20 sessions

### Execution convention
Signal generated from information known on date **T** → entry at the **next tradable session open** → future outcomes measured only from that executable entry.

### Required outputs
- `v3/labels/build_labels.py`
- `v3/data/labels_daily.parquet`
- `v3/data/model_dataset.parquet`

### Acceptance criteria
- Hand-check several known dates against raw market prices.
- Outcome-known dates are attached for all horizons.
- Labels never enter training before they would have become observable.

---

# Phase 3 — Establish simple model baselines

## V3-005 Logistic classification baseline

Predict:
- probability of positive 5-day return
- probability of positive 20-day return
- probability of positive 60-day return

Use regularization and chronological training only.

## V3-006 Regularized return regression baseline

Predict:
- expected 5-day return
- expected 20-day return
- expected 60-day return

## V3-007 Gradient-boosting candidate

Use a tree-boosting model appropriate for the small daily dataset. Avoid unnecessary deep-learning complexity.

## V3-008 Random-forest benchmark

Use primarily as a nonlinear comparison and robustness benchmark, not automatically as the champion.

### Common model interface
Every candidate must produce standardized prediction columns such as:
- `predicted_p_up_5d`
- `predicted_p_up_20d`
- `predicted_p_up_60d`
- `predicted_return_5d`
- `predicted_return_20d`
- `predicted_return_60d`
- `predicted_drawdown_20d`

---

# Phase 4 — Create the common walk-forward tournament

## V3-009 Build the evaluator

### Default chronological folds
- Train through 2023 → test 2024
- Train through 2024 → test 2025
- Train through 2025 → test 2026 YTD

Future folds should be appended without rewriting historical results.

### Metrics
#### Prediction quality
- Brier score
- log loss
- calibration error
- MAE/RMSE for expected returns and drawdowns
- rank correlation where appropriate

#### Trading usefulness
- total return
- annualized return
- benchmark-relative return
- Sharpe
- Sortino
- maximum drawdown
- Calmar
- turnover
- worst year
- time underwater

### Required outputs
- `v3/evaluation/walk_forward.py`
- `v3/evaluation/backtest.py`
- `v3/evaluation/metrics.py`

---

## V3-010 Build the model tournament scoreboard

Every candidate must be evaluated on exactly the same dates and assumptions.

Example columns:

| Experiment | Model | Features | 2024 | 2025 | 2026 YTD | Excess CAGR | Sharpe | Max DD | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Required outputs
- `v3/evaluation/tournament.py`
- `v3/reports/tournament.csv`
- `v3/reports/tournament.json`

### Rule
A candidate is not better because one headline metric improved. It must show robust improvement across multiple folds and risk metrics.

---

# Phase 5 — Add independent information one family at a time

## V3-011 Add VIX/volatility features

Candidate features:
- VIX level
- VIX change
- VIX percentile
- VIX relative to recent history
- term structure if reliable historical data are available

## V3-012 Run VIX ablation test

Compare identical models:
- without VIX
- with VIX

Keep VIX only if unseen performance improves consistently.

## V3-013 Add QQQ/SPY relative-strength features

Candidate features:
- QQQ returns
- QQQ/SPY relative return
- growth-vs-broad-market momentum
- divergence/reversal measures

## V3-014 Add market-breadth features

Candidate features:
- percentage above 20/50/200 DMA
- advance/decline measures
- new highs vs new lows

Only use a reliable point-in-time historical source.

## V3-015 Add macro/risk features

Candidate families:
- Treasury yields and changes
- high-yield credit spreads
- dollar/risk indicators

Each family requires its own ablation comparison.

---

# Phase 6 — Build the decision policy

## V3-016 Convert predictions into actions

The prediction engine and decision policy must remain separate.

Decision inputs should include quantities such as:
- expected 20-day return
- probability of positive 20-day return
- predicted 20-day drawdown
- probability of a further 5% decline
- model uncertainty/calibration quality

Initial action vocabulary:
- `STRONG ADD`
- `ADD MODESTLY`
- `BASELINE`
- `WAIT FOR BETTER ENTRY`

Do not introduce a sell/underweight action until evidence shows that reducing below baseline improves unseen portfolio results.

---

# Phase 7 — Add sizing only after prediction is proven

## V3-017 Test minimal sizing policy

Start with only:
- baseline = `1.00x`
- strongest validated positive signal = `1.10x`

Only if this works robustly out of sample should later experiments test `1.20x` or more.

### Do not initially test
- complex multi-tier leverage
- aggressive underweight exposure
- parameter sweeps designed to maximize backtest CAGR

---

# Phase 8 — Champion selection

## V3-018 Define champion acceptance gates

A v3 candidate should not replace v2.1 unless, after costs:

- benchmark-relative return is positive across the combined unseen periods
- improvement is not concentrated in one year
- Sharpe is at least as good as the benchmark and v2.1 baseline
- maximum drawdown is not materially worse without a justified return benefit
- calibration is acceptable
- no leakage/data-quality tests fail
- results remain reasonable under modest parameter/cost perturbations

Passing one metric is not enough.

## V3-019 Promote approved candidate

Model states:
- `candidate`
- `challenger`
- `champion`

Promotion must record:
- experiment ID
- model specification
- feature-set version
- label version
- training cutoff
- metrics
- artifact hashes

---

# Phase 9 — Dashboard integration

## V3-020 Integrate champion outputs into dashboard

The dashboard should display outputs, not contain model-training logic.

Target display:
- model/version
- latest data date
- expected 5/20/60-day return
- probability of positive return
- predicted downside
- probability of further material decline
- recommended action
- approved tactical multiplier
- validation status
- 2024 / 2025 / 2026 fold status
- v2.1 comparison

---

# Phase 10 — Untouched forward validation

## V3-021 Start immutable v3 live ledger

Every production candidate/champion prediction must be recorded before the outcome is known.

Store:
- prediction timestamp
- decision date
- model/version
- feature-set version
- model artifact hash
- inputs/data hash
- predictions
- action
- later realized outcomes

Never rewrite historical predictions after observing outcomes.

---

# Experiment organization

Every research run should receive an ID:

- `EXP-001`
- `EXP-002`
- `EXP-003`

Each experiment directory should contain:

```text
v3/experiments/EXP-XXX/
├── manifest.json
├── metrics.json
├── fold_metrics.csv
├── predictions.parquet
└── notes.md
```

The manifest should record at minimum:
- model type
- feature-set version
- label version
- training/test periods
- hyperparameters
- code commit SHA
- random seed
- data hash
- status

Never overwrite an old experiment to make its results look better.

---

# GitHub workflow

For each numbered task:

1. Open one issue (`V3-XXX`).
2. Create one focused branch.
3. Implement only that issue's scope.
4. Run tests/evaluation.
5. Open a PR with measured results.
6. Merge only if acceptance criteria pass.
7. Update this plan/status.

Avoid a months-long `v3-development` branch.

---

# Recommended implementation order

- [ ] V3-001 Freeze v2.1 baseline reports
- [ ] V3-002 Build research feature dataset
- [ ] V3-003 Add leakage/data-quality tests
- [ ] V3-004 Build multi-horizon labels
- [ ] V3-005 Logistic baseline
- [ ] V3-006 Return-regression baseline
- [ ] V3-007 Gradient boosting
- [ ] V3-008 Random-forest benchmark
- [ ] V3-009 Common walk-forward evaluator
- [ ] V3-010 Tournament scoreboard
- [ ] V3-011 Add VIX
- [ ] V3-012 VIX ablation
- [ ] V3-013 Add QQQ/SPY relative strength
- [ ] V3-014 Add breadth
- [ ] V3-015 Add macro/risk features
- [ ] V3-016 Decision policy
- [ ] V3-017 Minimal sizing test
- [ ] V3-018 Champion acceptance gates
- [ ] V3-019 Promote champion
- [ ] V3-020 Dashboard integration
- [ ] V3-021 Start v3 live ledger

---

# Immediate next step

Do **V3-001 through V3-004 first**. Do not start choosing ML winners until the frozen baseline, point-in-time feature table, leakage tests, and executable-entry labels are all correct and reproducible.
