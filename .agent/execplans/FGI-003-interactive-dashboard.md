# FGI-003 — Interactive, reproducible historical market explorer

## Purpose
Users can filter historical sentiment/strategy observations on a phone or desktop, see actual mature sample counts, inspect returns, and export the filtered published records; none of this alters operational or research decisions.

## Implemented locally
- [x] Sentiment time-series chart with 1M, 3M, 1Y, ALL ranges.
- [x] Historical date range, text search, action and regime selection, sorting, pagination and accessible table.
- [x] Descriptive sample count, mature-outcome count, positive share, average and median 5D return.
- [x] CSV export of only current selection with spreadsheet formula-escape rules.
- [x] Strategy methodology, caveats and V3 shadow percentile chart, without new model scoring.
- [x] Typed numeric/date parsing, null handling, empty data/invalid schema and stale warnings.
- [x] Node contract tests and Chromium desktop/mobile smoke tests authored.

## Required acceptance
- [ ] Real generated JSON/CSV test with non-empty and empty filters; count and outcome values manually spot-checked.
- [ ] Keyboard/ARIA and mobile touch review; Lighthouse/web-vitals, low-bandwidth and zoom behavior checked.
- [ ] Confirm outcome maturity semantics align with existing Python emitted data; no training leakage introduced.
- [ ] Public data source/license approval and owner-reviewed release.

## Next command
`bash "$HOME/Downloads/FearGreedIndex-react-production/tools/run_release_checks.sh"` and manual browser audit at `http://127.0.0.1:8000/`.

## Same-site release checkpoint (2026-10-09)

Interactive Overview, Explorer, Strategy and Research sections are implemented in the source package; verify real input schemas, matured outcomes, mobile/keyboard usability, exports and accessible charts before completing.
