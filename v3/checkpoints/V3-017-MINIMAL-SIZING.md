# V3-017 Minimal Sizing Checkpoint

## Status

**IMPLEMENTED — ACTIVATION BLOCKED UNTIL PREDICTION PROMOTION**

V3-017 implements the minimal sizing layer without violating the roadmap prerequisite that prediction edge must be proven first.

## Allowed sizing

- baseline: `1.00x`
- promotion-ready `STRONG ADD`: `1.10x`
- maximum allowed in V3-017: `1.10x`
- minimum allowed in V3-017: `1.00x`

No larger sizing and no underweight exposure are allowed.

## Current operational result

The current combined research candidate is **not promotion-ready**, therefore:

- current candidate multiplier: **1.00x**
- `STRONG ADD` extra sizing: **blocked**
- empirical 1.10x portfolio test: **deferred until prediction promotion gate passes**

## Artifacts

- `v3/policy/sizing_v1.json`
- `v3/policy/sizing_policy.py`
- `v3/reports/sizing_policy_manifest.json`
- `v3/reports/sizing_policy_result_PASS.txt`

The sizing manifest hashes both the config and engine code.

## Next

V3-018 must formalize and evaluate the champion acceptance gates. Only a model that passes those gates may unlock the V3-017 1.10x research sizing test.
