# V3 Diagnostics

This directory contains **diagnostic-only** research tools. Diagnostics explain dataset or model behavior; they do not select a champion, change the decision policy, or change sizing.

## Contract

- Diagnostics must use the same point-in-time and outcome-maturity rules as the V3 experiments they analyze.
- Every diagnostic checkpoint gets a GitHub issue with its scope frozen before its result is inspected.
- Preserve all in-scope rows/features rather than committing only interesting findings.
- Diagnostic findings may motivate a later experiment, but any feature/model/threshold decision requires a new pre-registered experiment ID.
- Compact diagnostic evidence belongs in `v3/reports/`; human-readable conclusions belong in `v3/checkpoints/`.
- Generated large/intermediate data remains rebuildable and out of Git unless explicitly required for reproducibility.

## Current diagnostic

`DIAG-001` measures target prevalence drift, feature distribution shift, and feature-target relationship/sign drift on the frozen 53-feature Treasury dataset and EXP-006 favorable-entry target.
