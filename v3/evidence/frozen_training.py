"""Immutable pre-cutoff STAB-004 training inputs for shadow replay.

The public dashboard uses continuously updated market observations; the frozen
STAB-004 shadow methodology must not silently retrain when a vendor revises
history. Keep the two source snapshots alongside this module under
``v3/frozen/STAB-004/``. The original input bytes are anchored to the Git
commit that first sealed the shadow predictions (2026-08-24).

No research outcomes are unsealed or written here. This module only reconstructs
historical features and labels for the pre-cutoff scoring path in memory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from v3.features.build_features import build_feature_frame, load_fear_greed, load_market as load_feature_market
from v3.features.build_treasury_features import build_expanded_features, load_source
from v3.labels.build_labels import build_labels, load_market as load_label_market

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = ROOT / "v3" / "frozen" / "STAB-004"
FROZEN_FEAR = SNAPSHOT_DIR / "fear_greed_daily.csv"
FROZEN_MARKET = SNAPSHOT_DIR / "spx_daily.csv"
FROZEN_INPUT_COMMIT = "d5d7b6c158dce96423298cb3fe3207265ec3f6f3"
# Git SHA-1 blob object IDs observed at the initial shadow-freeze commit.
FROZEN_SOURCE_BLOBS = {
    "fear_greed_daily.csv": "8f17037f5066819929df836933306844ea4820a3",
    "spx_daily.csv": "af8e7ccf704e3b7444064acedb42774f6ddb0fd8",
}
TREASURY = ROOT / "v3" / "data" / "treasury_daily.csv.gz"
FROZEN_TREASURY_SHA256 = "3691f934065c24003d9ea263f098cf4821800d777960e08771bd2037a3dc6d4a"


def git_blob_sha1(payload: bytes) -> str:
    """Compute the canonical Git SHA-1 *blob object* ID, not raw SHA-1."""
    return hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()


def verify_frozen_inputs() -> dict[str, str]:
    """Reject missing or revised training snapshots before any replay."""
    found = {}
    for path in (FROZEN_FEAR, FROZEN_MARKET):
        if not path.is_file():
            raise FileNotFoundError(
                f"Frozen STAB-004 source not installed: {path}. "
                "Run python extras/freeze_stab004_inputs.py and commit its new snapshots."
            )
        actual = git_blob_sha1(path.read_bytes())
        expected = FROZEN_SOURCE_BLOBS[path.name]
        if actual != expected:
            raise ValueError(
                f"Frozen STAB-004 source was modified: {path.name}; "
                f"expected Git blob {expected}, got {actual}. Do not revise frozen evidence."
            )
        found[path.name] = actual
    return found


def load_frozen_shadow_historical_dataset() -> pd.DataFrame:
    """Reconstruct original pre-cutoff training context without writing artifacts.

    Both raw historical source streams are frozen. Treasury history remains the
    separately frozen V3-015 snapshot. Model selection and ranking code remain
    untouched; this only prevents retrospective source revisions from changing
    the fitted shadow context for previously sealed decisions.
    """
    verify_frozen_inputs()
    # Both archived CSVs and the original gzip Treasury source are immutable.
    # Verify the original compressed bytes, not a potentially updated manifest.
    if not TREASURY.is_file():
        raise FileNotFoundError(f"Frozen Treasury snapshot is missing: {TREASURY}")
    treasury_digest = hashlib.sha256(TREASURY.read_bytes()).hexdigest()
    if treasury_digest != FROZEN_TREASURY_SHA256:
        raise ValueError(
            "Frozen Treasury snapshot does not match its original fingerprint; "
            f"got {treasury_digest}"
        )
    features = build_feature_frame(
        load_fear_greed(FROZEN_FEAR), load_feature_market(FROZEN_MARKET)
    )
    expanded = build_expanded_features(features, load_source(TREASURY))
    labels = build_labels(expanded["decision_date"], load_label_market(FROZEN_MARKET))
    return expanded.merge(labels, on="decision_date", how="left", validate="one_to_one")
