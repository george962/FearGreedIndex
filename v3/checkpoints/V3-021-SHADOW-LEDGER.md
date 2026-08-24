# V3-021 — Immutable shadow prediction ledger

## Status

**COMPLETE IMPLEMENTATION / RESEARCH-ONLY SHADOW INFRASTRUCTURE**

Issue: #76  
PR: #77  
Frozen challenger method: `STAB-004`  
Research cutoff: `2026-08-18`

This checkpoint records infrastructure only. It does not improve, retune, promote, or reinterpret STAB-004.

## Purpose

V3 challenger predictions must be preserved as they were generated before their future outcomes are known. The dashboard may display those predictions and compare them with the production v2.1 decision, but it may not reconstruct old challenger predictions after later research changes and present them as if they had been known at the time.

V3-021 therefore adds an append-only, hash-chained shadow prediction ledger downstream of the existing EVID-001 feature evidence lane.

## Initial immutable prediction chain

The first three predictions are bound to the already-sealed EVID-001 feature rows:

| Decision date | V3 opportunity rank | V3 state | Production action | Production timing |
| --- | ---: | --- | --- | --- |
| 2026-08-19 | 79.3651% | `ABSTAIN` | `HOLD / NO EXTRA BUYING` | `NEUTRAL / NO TIMING EDGE` |
| 2026-08-20 | 81.7460% | `STRONG_FAVORABLE` | `HOLD / NO EXTRA BUYING` | `NEUTRAL / NO TIMING EDGE` |
| 2026-08-21 | 84.9206% | `STRONG_FAVORABLE` | `HOLD / NO EXTRA BUYING` | `NEUTRAL / NO TIMING EDGE` |

All three predictions use 252 strictly prior raw-score references.

Prediction chain head after 2026-08-21:

`dd29baf39a69a93ec86040949141e993da99c0aa152855a76c17377cb8775779`

## Frozen representative set

The initial shadow predictions reproduce the frozen STAB-004 selector and redundancy-control rules and retain eight representatives:

- `spx_distance_ma_200`
- `spx_realized_vol_60`
- `spx_realized_vol_20`
- `spx_distance_high_252`
- `spx_return_60`
- `spx_realized_vol_5`
- `treasury_slope_change_20`
- `fg_distance_from_min_20`

The representative-set SHA-256 is:

`7ed2c5e8f62f74bb9810c71664770ea102ec98acfdc6932f341fc6e217c7fbab`

## Integrity contract

Each prediction row records and verifies:

- exact EVID-001 forward feature row SHA-256;
- feature-vector, source-feature, and feature-registry SHA-256 values;
- frozen STAB-004 evaluation and methodology-manifest hashes;
- representative-set hash and names;
- raw score, rolling percentile, reference count, and call state;
- `production_effect=NONE`;
- `sizing_multiplier=1`;
- previous/current prediction-row SHA-256 values.

The prediction ledger must remain a gap-free prefix of the collected EVID-001 feature ledger. Existing rows are immutable. If later code would recompute an older prediction differently, verification fails rather than rewriting history.

## Automation

The existing `Collect Untouched Forward Evidence` workflow now generates the point-in-time feature evidence and its matching research-only shadow prediction in one guarded run. It commits only the forward feature/source ledgers and the shadow prediction ledger, and refuses to publish if `main` changes during generation.

The Pages workflow also listens for successful completion of the forward-evidence workflow, then verifies the committed prediction lane before rendering it. This avoids relying on workflow-token push events to refresh the dashboard.

## Dashboard behavior

The challenger panel now reads the committed prediction ledger instead of recomputing historical V3 calls. It also writes `site/v3_challenger_history.csv` and displays recent production-vs-V3 decisions side by side.

The V3 renderer is downstream of the production dashboard build. PR CI hashes `site/analysis.json` before V3 rendering and verifies that the hash is identical afterward.

## Research / production boundary

- EVID-001 outcomes remain sealed.
- STAB-004 remains `complete_reject` and is not a champion.
- No V3-019 eligibility is created.
- Production BUY / WAIT / HOLD is unchanged.
- Production tactical sizing remains disabled at `1.00x`.
- The initial production/V3 disagreement is evidence to observe, not a reason to override production or retune the challenger.

## Next

Continue collecting immutable shadow predictions with the frozen method. Do not improve STAB/ADAPT until this infrastructure is merged, deployed, and verified. Subsequent methodology research should remain separate from this historical shadow lane so its past predictions stay auditable.
