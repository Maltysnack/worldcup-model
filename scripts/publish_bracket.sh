#!/usr/bin/env bash
# Copy the latest bracket.json + simulation.json into the maltysnack.github.io
# repo so the worldcup project page picks them up. Run after
# `uv run worldcup-bracket` and `uv run worldcup-simulate`.
#
# Usage: ./scripts/publish_bracket.sh

set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
DEST_DIR="$HOME/Documents/GitHub/Maltysnack.github.io/projects"

for name in bracket simulation; do
  src="$DATA_DIR/$name.json"
  dest="$DEST_DIR/worldcup-$name.json"
  if [[ ! -f "$src" ]]; then
    echo "ERROR: $src not found. Run 'uv run worldcup-$name' first." >&2
    exit 1
  fi
  cp "$src" "$dest"
  echo "Copied $src -> $dest"
done

echo "Now commit + push the maltysnack.github.io repo to update the live site."
