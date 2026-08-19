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

The integrity rerun rejects the previously assumed 76-feature combined configuration. Treasury is the only later feature family retained, with `UST-EXP-004` the best-ranked full candidate in that ablation. `UST-EXP-004` still fails the absolute prediction-promotion gate, therefore:

- current multiplier: **1.00x**
- `STRONG ADD` extra sizing: **blocked**
- empirical 1.10x portfolio test: **deferred until prediction promotion gate passes**

No rejected feature family may be used to bypass this gate.

## Artifacts

- `v3/policy/sizing_v1.json`
- `v3/policy/sizing_policy.py`
- `v3/reports/sizing_policy_manifest.json`
- `v3/reports/sizing_policy_result_PASS.txt`

The sizing manifest hashes both the config and engine code and is verified by the repository-integrity checkpoint.

## Next

V3-018 formalizes and evaluates fail-closed champion acceptance gates. Only a model that passes those gates may unlock the V3-017 1.10x research sizing test.
