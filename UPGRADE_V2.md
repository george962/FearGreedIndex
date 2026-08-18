# FearGreedIndex v2 validation upgrade

This package is intentionally validation-first.

It does **not** replace `scripts/build_dashboard.py`. That file remains the decision
engine and therefore the single source of truth for the dashboard, backtest,
walk-forward validation, and live signal ledger.

## Implemented replacements

- `backtest.py`
- `config.json`
- `.gitignore`
- `.github/workflows/market_data.yml`

## Implemented additions

- `strategy_manifest.json`
- `scripts/research_common.py`
- `scripts/strategy_validation.py`
- `scripts/signal_ledger.py`
- `test_strategy_validation.py`
- `test_signal_ledger.py`
- `.github/workflows/strategy_validation.yml`

## Local environment cleanup

```bash
rm -rf feargreed_env
```

The directory is no longer tracked, and `.gitignore` prevents it from being committed again.

## Local validation

```bash
python -m pip install -r requirements.txt

python -m unittest -v \
  test_feargreed.py \
  test_fear_greed_market_data.py \
  test_dashboard.py \
  test_strategy_validation.py \
  test_signal_ledger.py

python scripts/signal_ledger.py
python scripts/strategy_validation.py
python backtest.py
```

Outputs:

- `data/signal_ledger.csv`
- `reports/walk_forward_summary.csv`
- `reports/walk_forward_predictions.csv`
- `reports/walk_forward.json`
- `reports/backtest_daily.csv`
- `reports/decision_outcomes.csv`
- `reports/action_scorecard.csv`
- `reports/backtest_summary.json`

## What this upgrade fixes

### One decision engine

The old standalone backtest had a different date-alignment and return definition.
The replacement `backtest.py` calls the exact point-in-time dashboard engine.

### True validation separation

The strategy rules are frozen in `strategy_manifest.json`. The walk-forward
calibrator only uses observations prior to each test fold. The default folds are
2024, 2025, and 2026 YTD.

### Immutable live track record

Each current prediction gets a SHA-256 hash based on immutable decision fields.
If the same prediction is rerun, it is not duplicated. If source data or logic
changes the prediction, a new revision is appended rather than overwriting the
old one. Realized 1/5/10/20/60-session outcomes are filled only when available.

### Strategy-level metrics

The new backtest produces a transparent exposure-overlay simulation with
transaction-cost assumptions and reports total/annualized return, volatility,
Sharpe, Sortino, max drawdown, Calmar, turnover, and exposure.

## Important

Do not tune thresholds after looking at a failed holdout and then keep the same
`strategy_version`. Any threshold, regime, analog, sizing, or timing change should
bump the version and start a new validation record.

The next phase after this package is proven should be instrument-specific models
for SPY, QQQ, and TQQQ plus calibrated return/drawdown distributions displayed
directly on the dashboard.
