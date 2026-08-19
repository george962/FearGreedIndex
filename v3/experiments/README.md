# V3 Experiment Workspace

This directory contains **pre-registered research experiments** that happen after the staged V3 foundation roadmap.

## Repository roles

- `v3/PLAN.md` — original V3 development roadmap and permanent methodology rules.
- `v3/STATUS.md` — current source of truth for what is complete, rejected, retained, blocked, or active.
- `v3/experiments/EXP-XXX/` — experiment-specific pre-registration, immutable model/target contract, and final manifest.
- `v3/checkpoints/` — concise human-readable conclusions for completed milestones/experiments.
- `v3/reports/` — compact machine-readable evidence that is worth preserving in Git.
- `v3/data/` — source snapshots plus generated research datasets. Large generated Parquet files remain rebuildable and are not committed.
- `reports/baseline_v2_1/` — frozen operational benchmark. V3 research must not mutate it.

## Experiment lifecycle

Every new prediction experiment must follow this order:

1. Create a GitHub issue and assign a new immutable experiment ID.
2. Pre-register the target, input feature version, cutoff, chronological folds, model family/parameters, and pass/fail gate **before reading results**.
3. Implement the experiment on a branch from the latest stable `main` checkpoint.
4. Rebuild all required point-in-time inputs from a clean checkout.
5. Run the fixed experiment and preserve compact evidence.
6. Run the frozen-v2.1 and repository-integrity guards.
7. Record the conclusion under `v3/checkpoints/` and update `v3/STATUS.md`.
8. Merge both positive and negative experiments when their methodology is valid and reproducible.

A failed experiment is evidence, not a reason to tune the same experiment after seeing the result. Any material target, threshold, feature, or model change requires a new experiment/version.

## Active research direction

After V3-018 and EXP-005, the bottleneck is prediction quality rather than infrastructure. The current line of research tests **target reformulation first**, then regime conditioning only if the reformulated target demonstrates chronological out-of-sample learnability.
