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
- `shadow_prediction_ledger.csv` — append-only research-only challenger predictions bound to exact sealed feature rows.
- `shadow_predictions.py` — deterministic STAB-004 shadow collection, idempotency, semantic replay, and prediction-ledger hash-chain verification.
- `collect_forward_treasury.py` — captures new first-party FRED observations without modifying the frozen V3-015 Treasury snapshot.
- `append_forward_snapshot.py` — deterministic append/idempotency/hash-chain implementation.
- `collect_forward_evidence.py` — builds the latest point-in-time feature row and seals it; it never builds labels.
- `verify_forward_lane.py` — fail-closed integrity and governance validation for the feature/source evidence chains.

## Research rule

Collecting a feature snapshot does **not** expose its future outcome. Future outcomes may not be joined to this ledger, inspected for model selection, used for calibration, or used to alter feature/model rules before an explicit checkpoint is registered and intentionally opened.

Any predictive experiment designed after DIAG-001 must be pre-registered before it can consume outcomes from an untouched checkpoint. Opening a checkpoint makes that date range permanently research-exposed; it can never be called untouched again.

## Shadow prediction rule

A shadow prediction is allowed because it consumes only the already-frozen STAB-004 relationship contract plus the sealed point-in-time feature vector. It does **not** consume the future outcome of that decision date.

Each `shadow_prediction_ledger.csv` row records:

- the decision date and exact EVID-001 feature-row SHA-256;
- feature-vector/source/registry hashes;
- frozen STAB-004 evaluation and methodology-manifest hashes;
- the deterministic representative-set hash and representative names;
- raw score, causal rolling percentile, prior-score reference count, and call state;
- an explicit `production_effect=NONE` and `sizing_multiplier=1` guardrail;
- previous/current row SHA-256 values for an append-only chain.

The prediction ledger must be a gap-free prefix of the available forward feature ledger. Existing prediction rows are immutable: if a later code change would recompute an older row differently, collection and verification fail instead of rewriting history. This preserves what the challenger actually said before outcomes were known.

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

The shadow prediction ledger follows the same immutable-prefix principle and additionally replays every committed prediction from frozen code/data during verification. It may lag the feature lane briefly while automation is running, but it may never skip an earlier collected feature row and append a later prediction.

## Automation

`Collect Untouched Forward Evidence` runs after the existing market-data workflow succeeds. It is intentionally write-enabled only for the two append-only feature/source evidence ledgers. Before publishing, it verifies that remote `main` is still exactly the commit used to generate the evidence. If `main` moved during generation, the job refuses to rebase or push stale evidence and a later run must regenerate it.

`Collect V3 Shadow Predictions` runs after the untouched-forward workflow succeeds. It rebuilds the frozen V3 Treasury dataset, verifies EVID-001 is still sealed, appends any missing STAB-004 shadow predictions, semantically replays the whole committed prediction prefix, and commits **only** `shadow_prediction_ledger.csv`. It uses the same stale-generation guard before pushing.

The Pages dashboard reads the committed prediction ledger. It may show a temporary “awaiting immutable prediction” state when the feature collector has finished but the shadow collector has not yet committed the matching row. A later prediction-ledger commit triggers another Pages build.

## Promotion consequence

These lanes preserve future evidence and the challenger’s point-in-time predictions. They do not select a champion, enable V3-019, alter the production BUY/WAIT/HOLD action, or alter sizing. V3-017 remains 1.00x until a future candidate passes the existing champion gates on genuinely valid evidence.
