#!/usr/bin/env bash
# Copy the four worldcup JSON outputs into the maltysnack.github.io
# repo so the worldcup project page picks them up. Run after
# `uv run worldcup-bracket`, `uv run worldcup-simulate`,
# `uv run worldcup-build-live-tournament`, and `uv run worldcup-predict-top-scorer`.
#
# Usage: ./scripts/publish_bracket.sh
# Override site location with MALTYSNACK_SITE_DIR=/path/to/Maltysnack.github.io

set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
DEST_DIR="${MALTYSNACK_SITE_DIR:-$HOME/Documents/Maltysnack.github.io}/projects/worldcup"

if [[ ! -d "$DEST_DIR" ]]; then
  echo "ERROR: site worldcup dir not found: $DEST_DIR" >&2
  echo "Set MALTYSNACK_SITE_DIR=/path/to/Maltysnack.github.io to override." >&2
  exit 1
fi

# (worldcup source filename, site destination filename) — site uses the
# project-folder layout under projects/worldcup/ as of site commit e2cae15.
# live_tournament dropped May 2026 when the Live tab was removed.
PAIRS=(
  "bracket:bracket"
  "simulation:simulation"
  "top_scorers:top-scorers"
)

for pair in "${PAIRS[@]}"; do
  src_name="${pair%%:*}"
  dest_name="${pair##*:}"
  src="$DATA_DIR/${src_name}.json"
  dest="$DEST_DIR/${dest_name}.json"
  if [[ ! -f "$src" ]]; then
    echo "ERROR: $src not found. Run the matching worldcup-* command first." >&2
    exit 1
  fi
  cp "$src" "$dest"
  echo "Copied $src -> $dest"
done

echo
echo "Now commit + push the maltysnack.github.io repo to update the live site."
