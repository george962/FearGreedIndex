# FearGreedIndex v2.1 — Frozen Baseline

This folder is the version-level home for the current validated research baseline.

## Status

- Strategy version: `feargreed-v2.1.0`
- Role: frozen benchmark for v3 research
- Tactical sizing: disabled pending genuinely unseen validation
- Decision engine: current production/research implementation
- Live signal ledger: active

## Important

The v2.1 runtime files are intentionally still stored at the repository root for the moment. The existing GitHub Actions workflows, Pages deployment, data-update jobs, tests, and imports all depend on those paths. Moving the live files piecemeal through the GitHub API would create unnecessary risk of breaking the dashboard or scheduled data updates.

Treat the current root implementation as **v2.1 runtime code** until a dedicated migration PR moves it atomically and updates every workflow/import/test path together.

## Current v2.1 runtime map

| Area | Current path |
| --- | --- |
| Current Fear & Greed fetcher | `../FearGreed.py` |
| Historical Fear & Greed updater | `../FearGreedHistory.py` |
| SPX market-data builder | `../FearGreedMarketData.py` |
| Dashboard + decision engine | `../scripts/build_dashboard.py` |
| Shared research helpers | `../scripts/research_common.py` |
| Walk-forward validation | `../scripts/strategy_validation.py` |
| Signal ledger | `../scripts/signal_ledger.py` |
| Portfolio backtest | `../backtest.py` |
| Frozen configuration | `../config.json` |
| Strategy manifest | `../strategy_manifest.json` |
| Historical/live data | `../data/` |
| GitHub Actions | `../.github/workflows/` |
| Tests | repository-root `test_*.py` files |

## Freeze rule

Do not improve v2.1 by retuning thresholds after viewing later results. Any material change to features, model logic, analog matching, timing rules, validation methodology, or sizing belongs in v3 and should be evaluated against this baseline.

## What v2.1 is for now

1. Keep collecting the untouched forward signal ledger.
2. Keep the existing dashboard/data automation working.
3. Preserve v2.1 as the benchmark that v3 must beat.
4. Do not reactivate tactical sizing based only on retrospective tuning.

The v3 implementation plan is in [`../v3/PLAN.md`](../v3/PLAN.md).
