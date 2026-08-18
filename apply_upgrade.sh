#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-.}"

copy_file() {
  local rel="$1"
  mkdir -p "$TARGET_DIR/$(dirname "$rel")"
  cp "$SOURCE_DIR/$rel" "$TARGET_DIR/$rel"
  echo "updated $rel"
}

FILES=(
  "backtest.py"
  "config.json"
  ".gitignore"
  "strategy_manifest.json"
  "scripts/research_common.py"
  "scripts/strategy_validation.py"
  "scripts/signal_ledger.py"
  "test_strategy_validation.py"
  "test_signal_ledger.py"
  ".github/workflows/strategy_validation.yml"
  ".github/workflows/market_data.yml"
  "UPGRADE_V2.md"
  "reports/.gitkeep"
)

for file in "${FILES[@]}"; do
  copy_file "$file"
done

if [ -d "$TARGET_DIR/feargreed_env" ]; then
  echo
  echo "NOTE: feargreed_env is still present."
  echo "Run: git -C \"$TARGET_DIR\" rm -r feargreed_env"
fi

echo
echo "Upgrade files copied."
echo "Next run:"
echo "  cd \"$TARGET_DIR\""
echo "  python -m unittest -v test_feargreed.py test_fear_greed_market_data.py test_dashboard.py test_strategy_validation.py test_signal_ledger.py"
echo "  python scripts/signal_ledger.py"
echo "  python scripts/strategy_validation.py"
echo "  python backtest.py"
