# DATA-001 — Long-history core regime dataset

## Purpose

Increase independent market-regime coverage before another model or methodology iteration. This checkpoint is **data infrastructure only**: no model is fit, no CORE-001 result is inspected, and no production/champion/sizing state changes.

## Frozen source contract

- research window: `1990-01-02` through `2026-08-18`;
- S&P 500 daily OHLC: `^GSPC` through the repository's pinned `yfinance==0.2.65` acquisition path;
- VIX daily close: first-party Cboe historical VIX file;
- DGS2/DGS10: Federal Reserve H.15 series via FRED graph CSV endpoints;
- all three sources are frozen as deterministic checked-in gzip CSV snapshots with source/acquisition, normalized, and compressed snapshot hashes;
- downstream research reads the frozen snapshots and does not silently refresh them.

### Frozen snapshot evidence

| Source | Rows | Range | Frozen normalized SHA-256 | Frozen gzip SHA-256 |
| --- | ---: | --- | --- | --- |
| `^GSPC` | 9,224 | 1990-01-02 → 2026-08-18 | `fe4706b47fa5fec5e996e387593cea14bc58e8f9f7ae76a8a13774bf04f7299e` | `cb2e9ccc959edd3a1e5974207e0456dcb0a9261b17701b1f71c54a676259bc6d` |
| Cboe VIX | 9,253 | 1990-01-02 → 2026-08-18 | `8babdd0c33968b2b891f73f878b350d5fb63ed0a26f3d329cb9d7ba595179e33` | `3f0f3462a87e2ea4288dcb8c0835617d283e254a089de177289df0a7c787961c` |
| DGS2/DGS10 | 9,163 | 1990-01-02 → 2026-08-18 | `8eb61df4e2b10a966ad33b167cfea855dd2c01140ad5aa0488f57b2707ccb752` | `368ae200f4f78a344a4f8632ea7c2da8444b6a143c0d495352df78bce77fe5c2` |

Additional acquisition provenance is frozen in `v3/data/long_history/source_manifest.json`:

- SPX canonical yfinance acquisition-frame SHA-256: `912118e1737f861d775fa949a03b56e2ed56419ccdda2670fad0ee1a4f8b0ad4`;
- Cboe source-response SHA-256: `2cbdd7489a304eac0132205d4262083d4d6ae6208e16255db424ed5b5afee215`;
- DGS2 source-response SHA-256: `b9d9d6f9da18b51a1b8d37c9ed6e0fcff5d41f9d543e87e0a93b5c73ad76cb3a`;
- DGS10 source-response SHA-256: `33c564461d1038606014c62641d257ee7091c6f42297ebbf31e4f47dd9ad41c0`.

## Frozen feature registry

Feature set: `v3-long-history-core-001` — 47 features.

- 21 S&P 500 return/trend/high-low-position/realized-volatility features;
- 11 VIX level/change/return/percentile/moving-average-distance features;
- 15 Treasury 2Y/10Y/slope/change/percentile features;
- no CNN Fear & Greed fields;
- no cross-family interactions in DATA-001;
- VIX is joined to the exact decision session;
- Treasury is a backward-only as-of join and carries its source observation date for leakage validation.

## Label contract

Reuse the existing V3 executable-entry convention:

- decision on session `T`;
- entry at the **next tradable S&P 500 session open**;
- 5/20/60-session returns and path drawdowns preserve outcome-known dates;
- `favorable_entry_20d` matches EXP-006 exactly: forward return >= `+2%` and max drawdown strictly greater than `-5%`.

## Frozen validation result

The first end-to-end build passed all DATA-001 data gates:

- 9,224 decision-date feature rows;
- 47 fixed features;
- 8,009 rows complete across all 47 features;
- complete-data range: `1990-12-28` through `2026-08-18`;
- 9,204 mature `favorable_entry_20d` labels;
- complete rows by broad era: 1,564 in the 1990s, 2,264 in the 2000s, 2,516 in the 2010s, and 1,665 in the 2020s;
- source-date leakage validation: PASS;
- exact next-session-open label validation: PASS;
- Fear & Greed exclusion: PASS;
- `core_001_evaluated=false`;
- `champion_selected=false`, `v3_019_eligible=false`, sizing remains `1.00x`.

The generated Parquet feature/label/model datasets remain rebuildable artifacts. The source snapshots, source manifest, feature registry, compact validation summary, and this checkpoint are the frozen repository evidence.

## Acceptance gates

DATA-001 is accepted only if final-head CI proves:

1. frozen snapshot hashes reproduce exactly without another live refresh;
2. no source observation date is after its decision date;
3. the feature registry contains exactly 47 unique features and no Fear & Greed/outcome fields;
4. next-session-open labels reproduce the existing V3 convention;
5. at least 100 complete feature rows exist in each of the 1990s, 2000s, 2010s, and 2020s buckets;
6. the dataset remains infrastructure-only: `champion_selected=false`, `v3_019_eligible=false`, sizing `1.00x`, and `core_001_evaluated=false`.

## After DATA-001

Do **not** tune features from DATA-001 outcomes. Once the frozen-snapshot PR is final-head green and merged, create a separate pre-registration for `CORE-001` with broad chronological market-era folds **before** running predictive evaluation. Only after CORE-001 is frozen should `FG-INC-001` test the incremental value of audited-era Fear & Greed on identical dates/samples.
