#!/usr/bin/env bash
# Copy the latest bracket.json into the maltysnack.github.io repo so the
# worldcup project page picks it up. Run after `uv run worldcup-bracket`.
#
# Usage: ./scripts/publish_bracket.sh

set -euo pipefail

SRC="$(dirname "$0")/../data/bracket.json"
DEST="$HOME/Documents/GitHub/Maltysnack.github.io/projects/worldcup-bracket.json"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: $SRC not found. Run 'uv run worldcup-bracket' first." >&2
  exit 1
fi

cp "$SRC" "$DEST"
echo "Copied $SRC -> $DEST"
echo "Now commit + push the maltysnack.github.io repo to update the live site."
