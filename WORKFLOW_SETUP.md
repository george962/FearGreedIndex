# FearGreedIndex agent workflow and FGI-001 local delivery

This project owns its development workflow, defined in `.agent/WORKFLOW.md`. The planner, implementer, reviewer and verifier roles are responsibilities that Claude Code, Codex, or another coding agent can carry out. **No external workflow runner is required.** This ZIP includes documentation, tests and a local one-time presentation patch helper; it does not include a server, package manager or third-party agent framework.

## What FGI-001 adds

- A README that explains what a public visitor would do, how to reproduce a local build, and how to contribute safely.
- A more understandable overview of the existing dashboard, navigation to market context/history/methodology, keyboard skip links and CSS focus states.
- Separate, explicit sentiment observation age and site build time; research-only and not-real-time labeling.
- A user guide, provenance/licensing preflight, targeted presentation-contract tests.
- A project-owned `.agent/WORKFLOW.md`, revised agent rules, and a truthful partial checkpoint.

The change **does not** retune v2.1, promote v3, change data files, modify existing decision logic, enable indexing or deploy the site.

## Apply in your own local Git checkout

Do this from the `FearGreedIndex` directory on a clean, dedicated development branch. If you already have `feat/fgi-001-public-foundation`, switch to that branch instead of creating it again. Do not overwrite uncommitted work without inspecting the diff.

```bash
# If on main and no FGI-001 branch exists:
git switch main
git pull --ff-only origin main
git switch -c feat/fgi-001-public-foundation

# Extract the ZIP from Downloads into this directory, overwriting older
# versions of the same instruction files (inspect your Git diff afterward).
unzip -o ~/Downloads/FearGreedIndex-FGI001-owned-workflow.zip -d .

# Verify the original builder matches the safe presentation patch anchors.
python extras/apply_fgi001.py --check

# Apply presentation-only changes LOCALLY; this does not talk to GitHub.
python extras/apply_fgi001.py

git diff --stat
git diff --check

# Use the existing virtual environment if installed. From the repository root:
python -m unittest -v test_public_foundation
python -m unittest -v test_feargreed test_fear_greed_market_data test_dashboard test_signal_ledger test_strategy_validation test_v3_challenger
python -m unittest discover -s v3/tests -p 'test_*.py'
python scripts/build_dashboard.py
```

**Important:** Use `--check` first. The helper refuses to continue if a template anchor is missing/ambiguous or if onboarding was already added. If it fails, inspect `scripts/build_dashboard.py`; do not force the change. The helper is a delivery utility, not production runtime code. A complete copy of the modified dashboard script will exist in **your local checkout** after it succeeds.

Before applying presentation changes, capture a representative baseline build when possible and compare `site/analysis.json`, `site/historical_decisions.json`, `site/historical_decisions.csv` and `site/decision_changes.csv` after the change. Exclude intentionally variable build timestamps/IDs. **Do not change historical decisions to force parity.** Running the full build may require large data files, a configured source and local dependencies.

## Select and commit the actual project files

Review generated HTML with a local static file server and inspect accessibility, mobile display and warnings. If tests and semantic parity pass, commit the actual changes. You may keep `extras/apply_fgi001.py` untracked or remove it after applying; it is intentionally **not** required at runtime.

```bash
git add AGENTS.md CLAUDE.md DESIGN.md PRODUCT.md README.md WORKFLOW_SETUP.md \
  .agent docs/USER_GUIDE.md docs/PUBLICATION_PRECHECK.md \
  scripts/build_dashboard.py test_public_foundation.py

git commit -m "FGI-001: improve public research onboarding and agent workflow"
```

If you want to retain the existing documentation cleanup audit, it is included under `docs/REPOSITORY_CLEANUP.md` and may already be tracked. Do not delete test files, immutable evidence or root v2.1 runtime files for tidiness.

**No automatic GitHub push, merge, deployment or repository-settings change is performed.** You decide if/when to push the branch and open a PR. Update `.agent/STATUS.md` and FGI-001's ExecPlan with the actual branch, exit codes, diffs, findings and commit SHA(s) after local validation.

## Launch an agent in later sessions

> Read `AGENTS.md`, `.agent/WORKFLOW.md`, `.agent/STATUS.md` and the active ExecPlan. Continue the next approved unblocked step using planner → implementer → reviewer → verifier → checkpoint. Run actual tests, preserve frozen research outputs, update status, and commit to a feature branch. Stop before merge, deployment, publication or indexing.

## Outstanding FGI-001 blockers

- The public Pages URL and mobile/keyboard browser behavior have not been independently verified.
- Data source licensing, provenance and public redistribution permissions are not cleared.
- Search indexing, deployment and merging require your explicit approval.
- The full existing repository test suite and semantic parity checks could not be executed in this isolated package workspace; run them locally or in CI before declaring FGI-001 complete.
