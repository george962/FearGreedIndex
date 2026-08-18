#!/usr/bin/env python3
"""Generate the immutable v2.1 baseline package used by v3 comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "baseline_v2_1"
RUNTIME_FILES = (
    "FearGreed.py",
    "FearGreedHistory.py",
    "FearGreedMarketData.py",
    "backtest.py",
    "config.json",
    "strategy_manifest.json",
    "scripts/build_dashboard.py",
    "scripts/research_common.py",
    "scripts/strategy_validation.py",
)
REQUIRED_REPORTS = (
    "backtest_summary.json",
    "walk_forward_summary.csv",
    "action_scorecard.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    manifest = json.loads((ROOT / "strategy_manifest.json").read_text(encoding="utf-8"))
    version = str(manifest.get("strategy_version", "")).strip()
    if not version:
        raise SystemExit("strategy_manifest.json has no strategy_version")

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if config.get("enable_tactical_sizing") is not False:
        raise SystemExit(
            "V3-001 requires the frozen v2.1 benchmark to keep tactical sizing disabled"
        )

    with tempfile.TemporaryDirectory(prefix="feargreed-v2_1-") as temp_dir:
        temp = Path(temp_dir)

        # This is a benchmark capture, not a promotion gate. A frozen benchmark
        # must be recorded even if an existing v2.1 acceptance threshold reports
        # REVIEW/FAIL, so intentionally do not pass --strict here.
        run([
            sys.executable,
            "scripts/strategy_validation.py",
            "--skip-yahoo-fallback",
            "--output-dir",
            str(temp),
            "--progress-every",
            "100",
        ])
        run([
            sys.executable,
            "backtest.py",
            "--skip-yahoo-fallback",
            "--output-dir",
            str(temp),
            "--progress-every",
            "100",
        ])

        missing = [name for name in REQUIRED_REPORTS if not (temp / name).exists()]
        if missing:
            raise SystemExit(f"Baseline generation missing outputs: {missing}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_REPORTS:
            shutil.copy2(temp / name, args.output_dir / name)

        backtest_summary = json.loads((temp / "backtest_summary.json").read_text(encoding="utf-8"))
        file_hashes = {relative: sha256(ROOT / relative) for relative in RUNTIME_FILES}
        input_hashes = {
            "data/fear_greed_daily.csv": sha256(ROOT / "data" / "fear_greed_daily.csv"),
            "data/spx_daily.csv": sha256(ROOT / "data" / "spx_daily.csv"),
        }
        report_hashes = {name: sha256(args.output_dir / name) for name in REQUIRED_REPORTS}

        freeze_manifest = {
            "strategy_version": version,
            "dataset_start": backtest_summary.get("start"),
            "dataset_end": backtest_summary.get("end"),
            "tactical_sizing_enabled": False,
            "runtime_sha256": file_hashes,
            "input_sha256": input_hashes,
            "report_sha256": report_hashes,
            "generation_command": "python v3/baseline/freeze_v2_1.py",
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(freeze_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        readme = (
            f"# v2.1 Frozen Baseline\n\n"
            f"Strategy version: `{version}`\n\n"
            "This directory is the permanent benchmark package for v3 research. "
            "It is generated exclusively from checked-in data using the frozen v2.1 runtime.\n\n"
            f"Evaluation coverage: `{backtest_summary.get('start')}` through "
            f"`{backtest_summary.get('end')}`.\n\n"
            "Tactical sizing is explicitly required to remain disabled while this "
            "benchmark is generated.\n\n"
            "## Reproduce\n\n```bash\npython v3/baseline/freeze_v2_1.py\n```\n\n"
            "The hashes in `manifest.json` identify the exact runtime, inputs, and reports. "
            "Any methodology change belongs in v3 rather than changing this benchmark.\n"
        )
        (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Frozen v2.1 baseline written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
