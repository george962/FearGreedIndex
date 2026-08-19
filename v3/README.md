# FearGreedIndex v3 — Research Workspace

V3 is the next-generation predictive research system. It is intentionally separated from the frozen v2.1 operational benchmark so experimental work cannot silently change the live strategy.

## Start here

Use these files for different purposes:

- [`PLAN.md`](PLAN.md) — original staged V3 roadmap and permanent methodology rules.
- [`STATUS.md`](STATUS.md) — **current source of truth** for completed, retained, rejected, blocked, and active work.
- [`experiments/README.md`](experiments/README.md) — post-roadmap experiment lifecycle and repository organization rules.
- `experiments/EXP-XXX/` — pre-registration and immutable experiment contract.
- `checkpoints/` — concise human-readable conclusions from completed milestones/experiments.
- `reports/` — compact machine-readable evidence worth preserving in Git.

Do not infer the current research stage from an old branch or an individual report; read `STATUS.md`.

## Current research state

- v2.1 remains frozen and operationally unchanged.
- V3-001 through V3-018 are implemented, with V3-019 blocked because no candidate passes the champion gates.
- Treasury/yield-curve context is the only later feature family retained by the repaired same-sample ablations.
- VIX, QQQ/SPY relative strength, broad-dollar, and the combined feature stack were rejected under their frozen gates.
- EXP-005 improved probability calibration but still failed the absolute prediction prerequisite.
- Active research therefore returns to the **prediction formulation**, not sizing or champion promotion.
- EXP-006 tests whether a 20-session entry-opportunity target is more learnable than exact forward-return/drawdown prediction.

See `STATUS.md` and `experiments/EXP-006/PLAN.md` for exact frozen decisions and active criteria.

## Point-in-time contract

- `decision_date` is the date on which the feature vector is known.
- Fear & Greed values are backward-as-of joined; future source dates are forbidden.
- Market/macro features use observations available on or before the decision date under their declared availability convention.
- Labels remain separate from features.
- A decision on date T enters at the next tradable session open.
- 5/20/60-session outcomes include the entry session as session 1.
- `_forward_*_known_date` fields state when each outcome becomes legally available to training.
- Training eligibility is controlled by the outcome-known date, not merely by `decision_date`.

## Core reproducibility commands

```bash
python -m v3.features.build_features
python -m v3.labels.build_labels
python -m v3.evaluation.validate_dataset
python -m v3.baseline.freeze_v2_1
python -m unittest discover -s v3/tests -p 'test_*.py'
```

Feature-family and experiment workflows rebuild any additional required snapshots/datasets before evaluation.

## Artifact policy

Generated Parquet prediction/model datasets are rebuildable and normally remain out of Git. Compact immutable evidence, manifests, checkpoints, and source-snapshot metadata are committed when they are necessary to reproduce a research conclusion.

Every positive **and negative** experiment is preserved. A failed result cannot be retroactively tuned under the same experiment ID; material changes require a new pre-registered experiment/version.
