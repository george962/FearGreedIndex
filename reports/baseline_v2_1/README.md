# v2.1 Frozen Baseline

Strategy version: `feargreed-v2.1.0`

This directory is the permanent benchmark package for v3 research. It is generated exclusively from checked-in data using the frozen v2.1 runtime.

Evaluation coverage: `2021-02-02` through `2026-08-18`.

The recorded dataset end is an immutable cutoff. Later live-data appends are excluded from replay, outcomes, and input fingerprints. Input hashes cover the parsed fields actually consumed by v2.1 through that cutoff.

Tactical sizing is explicitly required to remain disabled while this benchmark is generated. The walk-forward status is recorded as evidence, not used as a condition for whether the benchmark may be frozen.

## Reproduce

```bash
python v3/baseline/freeze_v2_1.py
```

The hashes in `manifest.json` identify the exact runtime, frozen input sample, generator, and reports. Any methodology change belongs in v3 rather than changing this benchmark.
