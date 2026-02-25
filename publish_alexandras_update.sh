#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/Users/troboukis/Code/alexandras"
SITE_REPO="/Users/troboukis/Code/troboukis.github.io"
SITE_SUBDIR="$SITE_REPO/alexandras"

COMMIT_MSG="${1:-alexandras: update index and data ($(date '+%Y-%m-%d %H:%M:%S'))}"

echo "Step 1/4: Updating Alexandras data..."
cd "$SRC_DIR"
python update_alexandras_data.py --log-file logs/manual_run.log --verbose

echo "Step 2/4: Copying files to site repo..."
mkdir -p "$SITE_SUBDIR"
cp "$SRC_DIR/index.html" "$SITE_SUBDIR/index.html"
cp "$SRC_DIR/data_alexandras.csv" "$SITE_SUBDIR/data_alexandras.csv"

echo "Step 3/4: Switching to site repo..."
cd "$SITE_REPO"

echo "Step 4/4: Git add/commit/push (only alexandras files)..."
git add alexandras/index.html alexandras/data_alexandras.csv

if git diff --cached --quiet -- alexandras/index.html alexandras/data_alexandras.csv; then
  echo "No changes to commit for alexandras/index.html or alexandras/data_alexandras.csv"
  exit 0
fi

git commit -m "$COMMIT_MSG" -- alexandras/index.html alexandras/data_alexandras.csv
git push

echo "Publish complete."
