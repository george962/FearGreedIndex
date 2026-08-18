# FearGreedIndex v3 — Research Workspace

v3 is the next-generation predictive research system. It is intentionally separated from the frozen v2.1 baseline so experimental work cannot silently change the current operational strategy.

## Start here

Read [`PLAN.md`](PLAN.md) for the authoritative staged implementation plan, acceptance gates, experiment rules, and recommended order of work.

## Foundation pipeline — V3-001 through V3-004

```bash
python v3/baseline/freeze_v2_1.py
python v3/features/build_features.py
python v3/labels/build_labels.py
python v3/evaluation/validate_dataset.py
python -m unittest -v \
  v3.tests.test_features \
  v3.tests.test_labels \
  v3.tests.test_leakage
```

### Point-in-time contract

- `decision_date` is the date on which the feature vector is known.
- Fear & Greed values are joined backward-as-of; a future source date is forbidden.
- Market features use the current or earlier market observations only.
- Labels remain separate from features.
- A decision on date T enters at the next tradable session open.
- 5/20/60-session targets include the entry session as session 1.
- `_forward_*_known_date` fields state when each outcome becomes legally available to training.
- Model training must gate rows by the applicable known date, not merely by `decision_date`.

## Generated artifacts

The foundation pipeline produces:

- `reports/baseline_v2_1/backtest_summary.json`
- `reports/baseline_v2_1/walk_forward_summary.csv`
- `reports/baseline_v2_1/action_scorecard.csv`
- `reports/baseline_v2_1/manifest.json`
- `v3/data/features_daily.parquet`
- `v3/data/labels_daily.parquet`
- `v3/data/model_dataset.parquet`
- `v3/reports/features_missingness.json`
- `v3/reports/dataset_validation.json`

The generated Parquet/report artifacts can be rebuilt from checked-in data and code. The v2.1 baseline manifest records SHA-256 hashes for the frozen runtime, inputs, and reports.

## Current status

- v2.1 remains the frozen benchmark.
- v3 has not been promoted to production.
- V3-001 through V3-004 are the only active implementation scope until the foundation CI gate passes.
- No model tournament or tactical sizing changes should begin until the foundation is reproducible and leakage validation passes.
