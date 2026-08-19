# FearGreedIndex v3 — Research Workspace

V3 is the next-generation predictive research system. It is intentionally separated from the frozen v2.1 operational benchmark so experimental work cannot silently change the live strategy.

## Start here

Use these locations for distinct purposes:

- [`PLAN.md`](PLAN.md) — original staged V3 roadmap and permanent methodology rules.
- [`STATUS.md`](STATUS.md) — **current source of truth** for completed, retained, rejected, blocked, and active work.
- [`experiments/README.md`](experiments/README.md) — predictive experiment lifecycle and immutability rules.
- `experiments/EXP-XXX/` — pre-registration and frozen experiment contracts.
- [`diagnostics/README.md`](diagnostics/README.md) — diagnostic-only research rules; diagnostics explain behavior but do not select models.
- `checkpoints/` — concise human-readable conclusions and historical repair records.
- `reports/` — compact machine-readable evidence worth preserving in Git.

Do not infer current research state from an old branch or individual report; read `STATUS.md`.

## Current research state

- v2.1 remains frozen and operationally unchanged.
- V3-001 through V3-018 are implemented; V3-019 remains blocked because no candidate passes the champion gates.
- Treasury/yield-curve context is the only later feature family retained by repaired same-sample ablations.
- VIX, QQQ/SPY relative strength, broad dollar, and the combined feature stack were rejected under their frozen gates.
- EXP-005 through EXP-009 are completed negative experiments. They tested calibration, target reformulation, a simple rate regime, sentiment extremes, and fixed recent-window adaptation without producing promotion-ready predictive edge.
- DIAG-001 is complete and identifies broad conditional relationship drift: 42/53 features reverse training-to-test association sign at least once and 34/53 change test sign across research years. The Fear & Greed family is especially unstable.
- Because DIAG-001 inspected 2024–2026 YTD outcomes, those periods are now development/research evidence for post-DIAG hypotheses, not fresh final promotion evidence.
- **Next work is governance + methodology:** establish an untouched forward-evidence lane after the `2026-08-18` research cutoff, then test a past-only stability-selection methodology under a new pre-registration.
- No champion is selected; V3-019 is blocked and sizing remains `1.00x`.

See `STATUS.md` and `checkpoints/DIAG-001.md` for the current evidence and next-step constraints.

## Point-in-time contract

- `decision_date` is the date on which the feature vector is known.
- Fear & Greed values are backward-as-of joined; future source dates are forbidden.
- Market/macro features use observations available on or before the decision date under their declared availability convention.
- Labels remain separate from features.
- A decision on date T enters at the next tradable session open.
- 5/20/60-session outcomes include the entry session as session 1.
- `_forward_*_known_date` fields state when each outcome becomes legally available to training.
- Training eligibility is controlled by the outcome-known date, not merely by `decision_date`.

## Evidence-use contract after DIAG-001

- Research outcomes through `2026-08-18` have been inspected and may be used for development diagnostics.
- Any feature/model rule motivated by DIAG-001 must use a new pre-registered experiment ID.
- Historical outer-fold results from 2024–2026 YTD must not be described as untouched promotion evidence for a post-DIAG model.
- Final promotion requires fresh evidence that did not participate in formulation, such as an explicit forward holdout after the frozen research cutoff or separately acquired unused historical data.

## Core reproducibility commands

```bash
python -m v3.features.build_features
python -m v3.labels.build_labels
python -m v3.evaluation.validate_dataset
python -m v3.baseline.freeze_v2_1
python -m unittest discover -s v3/tests -p 'test_*.py'
```

Feature-family, experiment, and diagnostic workflows rebuild their additional required datasets before evaluation.

## Artifact policy

Generated Parquet predictions/model datasets are rebuildable and normally remain out of Git. Compact immutable evidence, manifests, checkpoints, diagnostics, and source-snapshot metadata are committed when necessary to reproduce a research conclusion.

Every positive **and negative** experiment is preserved. A failed result cannot be retroactively tuned under the same experiment ID; material changes require a new pre-registered experiment/version. Diagnostic findings likewise require a new pre-registered experiment before they can change model/feature choices.
