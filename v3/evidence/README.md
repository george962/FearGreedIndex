# V3 untouched forward evidence

This directory is the evidence boundary for research designed after DIAG-001.

DIAG-001 inspected 2024 through 2026 YTD outcomes, so those dates are research-exposed for any hypothesis informed by that diagnostic. They remain useful for development, but they are not fresh promotion evidence.

## Forward boundary

- Research-exposed through: **2026-08-18**.
- First eligible untouched decision date: **2026-08-19**.
- `forward_feature_ledger.csv` may contain point-in-time feature inputs and provenance only.
- It must never contain forward returns, drawdowns, labels, realized performance, or outcome-known dates.
- Missed decision dates are left as gaps. The collector never reconstructs/backfills an allegedly untouched row later.

## Files

- `forward_lane_manifest.json` — immutable lane contract and schema/version policy.
- `forward_feature_ledger.csv` — append-only sealed feature snapshots.
- `forward_treasury_source.csv` — append-only post-cutoff DGS2/DGS10 observations with the date on which they were first captured.
- `forward_checkpoints.json` — explicit evaluation/unseal registry. It starts empty.
- `collect_forward_treasury.py` — captures new first-party FRED observations without modifying the frozen V3-015 Treasury snapshot.
- `append_forward_snapshot.py` — deterministic append/idempotency/hash-chain implementation.
- `collect_forward_evidence.py` — builds the latest point-in-time feature row and seals it; it never builds labels.
- `verify_forward_lane.py` — fail-closed integrity and governance validation for both hash chains.

## Research rule

Collecting a feature snapshot does **not** expose its future outcome. Future outcomes may not be joined to this ledger, inspected for model selection, used for calibration, or used to alter feature/model rules before an explicit checkpoint is registered and intentionally opened.

Any predictive experiment designed after DIAG-001 must be pre-registered before it can consume outcomes from an untouched checkpoint. Opening a checkpoint makes that date range permanently research-exposed; it can never be called untouched again.

## Point-in-time Treasury rule

The checked-in V3-015 Treasury research snapshot remains frozen. EVID-001 never refreshes it.

New DGS2/DGS10 observations are written to `forward_treasury_source.csv` with `captured_on_date`. A decision-date snapshot may use only observations whose `captured_on_date <= decision_date`. This prevents a later FRED release or revision from being silently treated as if it had been known earlier.

## Ledger behavior

The feature ledger stores the complete registered 53-feature vector as canonical strings so its contents are reproducible across CSV round-trips. Every row also records:

- decision date and source dates;
- feature-set version and registry SHA-256;
- ordered feature-vector SHA-256;
- source-snapshot SHA-256;
- previous row SHA-256 and current row SHA-256.

The Treasury source ledger has its own chain and raw-source hashes. Appending the same feature decision date with the same snapshot is idempotent. Reusing a date with different evidence, inserting an older date after a newer row, changing an existing row, or including a forbidden outcome field is a hard failure.

## Automation

`Collect Untouched Forward Evidence` runs after the existing market-data workflow succeeds. It is intentionally write-enabled only for the two append-only evidence ledgers. Before publishing, it verifies that remote `main` is still exactly the commit used to generate the evidence. If `main` moved during generation, the job refuses to rebase or push stale evidence and a later run must regenerate it.

## Promotion consequence

The lane only preserves future evidence. It does not select a champion, enable V3-019, or alter sizing. V3-017 remains 1.00x until a future candidate passes the existing champion gates on genuinely valid evidence.
