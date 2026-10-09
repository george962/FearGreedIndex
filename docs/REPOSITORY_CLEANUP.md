# Repository cleanup audit — FearGreedIndex

Reviewed against the repository's `main` file inventory and documented workflows on 2026-10-09. **This is a proposal, not a verified safe-to-delete list. No files have been deleted.**

## Immediate deletions

**None recommended without a dependency/retention audit.** Test files are part of the release safety net; deleting them is the opposite of becoming production-ready. The tracked `site/.gitkeep` and `reports/.gitkeep` are tiny placeholders, not a meaningful cleanup opportunity.

## Review/possibly archive in a dedicated hygiene PR

| Path | Reason to review | Delete only after |
| --- | --- | --- |
| `fearandgreed.ipynb` | Original exploratory notebook may overlap with scripted production analysis | Confirm no unique findings/code, no workflow/docs references and whether preserving it in `examples/` or `archive/` makes more sense |
| `README-dashboard.md` | Documentation partly duplicates README and v2.1 docs | Consolidate unique install, deployment, data-source, security and limitation guidance into a single docs location; preserve links or add redirects |
| `site/.gitkeep` | Placeholder in generated output directory | Confirm directory is generated reliably and Git tracking of the empty directory is no longer useful |
| `reports/.gitkeep` | Placeholder, minimal value | Verify reports directory creation is automatic and retained reports are unaffected |

## KEEP: operational and test dependencies

- `FearGreed.py`, `FearGreedHistory.py`, `FearGreedMarketData.py`, `backtest.py`, `track_position.py`, `http_retry.py`, `scripts/`, `config.json`, `strategy_manifest.json`, `positions.json`, `requirements.txt`: treat as operational/research or potentially referenced until an exhaustive reference audit establishes otherwise. `positions.json` needs a *privacy review* before any public publishing, not an automatic deletion.
- Every root `test_*.py` file (including `test_backtest.py`, `test_dashboard.py`, `test_signal_ledger.py`, `test_v3_challenger.py`) and `v3/tests/`: keep. At least some are explicitly invoked by GitHub Actions; others guard behaviors that may not be covered by the deploy workflow's shorter test list.
- `.github/workflows/`: keep all scheduled ingestion, forward collection, CI and deployment workflows until a replacement is proven. Some newer workflows are task-specific validation gates; do not remove just because they contain a version number.
- `data/signal_ledger.csv`, `v3/evidence/`, `v3/checkpoints/`, baseline and experiment reports: immutable/append-only research evidence. Never delete/rewrite for tidiness.
- `v2.1/`, `v3/`, `data/`, `reports/baseline_v2_1/`: documented research and reproducibility boundaries.

## Safe cleanup process

1. On a separate `chore/repository-hygiene` branch, inventory `git ls-files`, repository references with `git grep`, CI workflow paths, imports (including dynamic imports), generated outputs and user-facing links.
2. Classify each candidate as `KEEP`, `REFACTOR`, `ARCHIVE`, or `DELETE`; write the justification and recovery path in the cleanup PR. Treat ambiguous files as `KEEP`.
3. For documentation, migrate unique content and update links before removing older pages. For notebooks, preserve any unique evidence or examples. For code, add tests covering replacement behavior before removing it.
4. Run all root and v3 tests, static dashboard build, the same deployment checks CI performs, and any targeted scripts/tests for the changed paths. Do not open sealed forward outcomes to test cleanup.
5. Require explicit owner approval for research/data deletion, public-data changes and any history rewrite. Merge only after CI and a manual diff review.

Avoid deleting Python tests simply to make the repository appear smaller. Production readiness comes from a stable structure and verified behavior, not fewer files.
