# V3 integrity repair

This branch repairs repository-state problems discovered during the V3-018 resume audit.

## Problems confirmed on `main`

1. V3-013 QQQ/SPY relative-strength assets were never merged even though later combined-feature code imports them.
2. `v3/STATUS.md` and `v3/ci/run_current_stage.py` are stale at V3-012/V3-013-era state.
3. Temporary write-enabled `v3_*_finalize.yml` workflows leaked onto `main`.
4. Multiple checkpoints claim immutable report/manifest files that are missing from `main` because finalizer-generated files were not included in the merge commits.

## Repair rule

Do not trust prior prose-only KEEP/PASS claims. Rebuild the affected frozen experiments under current code and preserve the generated evidence before updating status or resuming champion selection.

The repair covers V3-013, V3-015A/B/C, V3-016, and V3-017. V3-014 remains data-source blocked.
