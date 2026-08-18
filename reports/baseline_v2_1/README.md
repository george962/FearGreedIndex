# v2.1 Frozen Baseline

This directory is populated by `python v3/baseline/freeze_v2_1.py`.

The generator runs the frozen v2.1 walk-forward validation and portfolio backtest using only checked-in data, then writes:

- `backtest_summary.json`
- `walk_forward_summary.csv`
- `action_scorecard.csv`
- `manifest.json` with runtime/input/report SHA-256 hashes

Do not hand-edit generated baseline reports. Any methodology change belongs in v3.
