# V3-016 Decision Policy Checkpoint

## Status

**COMPLETE — RESEARCH-ONLY POLICY FRAMEWORK**

V3-016 implements a deterministic layer that converts standardized model predictions into an action without training models, sizing positions, selling, underweighting, or placing orders.

## Action vocabulary

- `STRONG ADD`
- `ADD MODESTLY`
- `BASELINE`
- `WAIT FOR BETTER ENTRY`

`WAIT FOR BETTER ENTRY` means maintain the baseline allocation and make no extra add. It is not a sell or underweight instruction.

## Frozen policy artifact

- Policy version: `v3-decision-policy-001`
- Status: `research_only`
- Config: `v3/policy/policy_v1.json`
- Engine: `v3/policy/decision_policy.py`
- Immutable manifest: `v3/reports/decision_policy_manifest.json`

The manifest records SHA-256 hashes for both policy config and engine code and is verified by the repository-integrity checkpoint.

## Safety / separation guarantees

- no position multiplier in V3-016
- no exposure sizing in V3-016
- no sell/underweight action
- low calibration or excessive uncertainty falls back to `BASELINE`
- output preserves model/feature/label/training/prediction lineage
- machine-readable reason codes are emitted
- same input + same policy produces the same result

## Validation

The policy tests cover all four action paths, confidence degradation, inclusive threshold boundaries, invalid probabilities, deterministic output, exact action vocabulary, and absence of hidden sizing/sell semantics.

## Corrected research-candidate context

The integrity rerun invalidated the earlier assumption that the 76-feature combined stack was retained. QQQ/SPY, broad dollar, and the combined stack are rejected under their frozen retention rules. Treasury is the only later feature family that remains retained, and `UST-EXP-004` is the best-ranked full candidate in the Treasury ablation. It is still **not promotion-ready**.

The decision policy remains deliberately model-agnostic and unbound to a production champion.

## Next

V3-017 keeps sizing separate:

- baseline = `1.00x`
- strongest promotion-ready positive signal = `1.10x`

V3-018/V3-019 must establish champion status before any extra sizing can activate.
