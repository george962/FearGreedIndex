#!/usr/bin/env python3
"""Run the current lightweight V3 repository checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v3.ci.check_repository_integrity import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
